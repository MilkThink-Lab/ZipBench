import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.zipbench.naming import zip_file_key
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{zip_file_key(dataset)}_supp.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        pred_format = get_pred_file_format()
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.{pred_format}'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False,
               batch_size=1):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{zip_file_key(dataset)}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen3-VL' in model_name
        or 'InternVL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}
        if batch_size > 1:
            kwargs['batch_size'] = batch_size

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    use_batch = batch_size > 1 and getattr(model, 'use_vllm', False)
    if use_batch:
        # Pre-build all pending structs, then submit them in chunks of `batch_size`
        # so vLLM can exploit continuous batching. Checkpoint after each chunk.
        structs, struct_idx = [], []
        for i in range(lt):
            idx = data.iloc[i]['index']
            if idx in res:
                continue
            if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
                struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
            else:
                struct = dataset.build_prompt(data.iloc[i])
            structs.append(struct)
            struct_idx.append(idx)

        for start in tqdm(
            range(0, len(structs), batch_size),
            desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size} (bs={batch_size})'
        ):
            chunk = structs[start:start + batch_size]
            chunk_idx = struct_idx[start:start + batch_size]
            try:
                responses = model.generate_batch(chunk, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.warn(f'Batched generation failed ({type(err)} {str(err)}), '
                              f'falling back to per-sample generation for this chunk')
                responses = []
                for struct in chunk:
                    if os.environ.get('SKIP_ERR', False) == '1':
                        FAIL_MSG = 'Failed to obtain answer'
                        try:
                            responses.append(model.generate(message=struct, dataset=dataset_name))
                        except RuntimeError as err2:
                            torch.cuda.synchronize()
                            warnings.warn(f'{type(err2)} {str(err2)}')
                            responses.append(f'{FAIL_MSG}: {type(err2)} {str(err2)}')
                    else:
                        responses.append(model.generate(message=struct, dataset=dataset_name))
            torch.cuda.empty_cache()

            for idx, response in zip(chunk_idx, responses):
                if verbose:
                    print(response, flush=True)
                res[idx] = response
            dump(res, out_file)
    else:
        for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
            idx = data.iloc[i]['index']
            if idx in res:
                continue

            if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
                struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
            else:
                struct = dataset.build_prompt(data.iloc[i])

            # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
            if os.environ.get('SKIP_ERR', False) == '1':
                FAIL_MSG = 'Failed to obtain answer'
                try:
                    response = model.generate(message=struct, dataset=dataset_name)
                except RuntimeError as err:
                    torch.cuda.synchronize()
                    warnings.warn(f'{type(err)} {str(err)}')
                    response = f'{FAIL_MSG}: {type(err)} {str(err)}'
            else:
                response = model.generate(message=struct, dataset=dataset_name)
            torch.cuda.empty_cache()

            if verbose:
                print(response, flush=True)

            res[idx] = response
            if (i + 1) % 10 == 0:
                dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# Add for agent evaluation
def _is_structured_record(v):
    return isinstance(v, dict) and 'prediction' in v and 'extra_records' in v


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, use_vllm=False,
    batch_size=1
):
    rank, world_size = get_rank_and_world_size()
    # ZipBench subsets get their own file names so full/subset runs never collide
    file_key = zip_file_key(dataset)
    # 使用环境变量控制的文件格式
    result_file = get_pred_file_path(work_dir, model_name, file_key, use_env_format=True)

    prev_file = f'{work_dir}/{model_name}_{file_key}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            # breakpoint()
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{file_key}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm,
        batch_size=batch_size)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        if os.getenv('SPLIT_THINK', False):
            if all(_is_structured_record(data_all[x]) for x in data['index']):
                prediction = [data_all[x]['prediction'] for x in data['index']]
                extra_records = [data_all[x]['extra_records'] for x in data['index']]
                data['extra_records'] = extra_records
            else:
                prediction = [str(data_all[x]) for x in data['index']]

            def split_thinking(s):
                if '</think>' in s:
                    splits = s.split('</think>')
                    prediction = splits[-1].strip()
                    if len(splits) == 2 and '<think>' in splits[0]:
                        thinking = splits[0].split('<think>')[1].strip()
                    else:
                        thinking = '</think>'.join(splits[:-1])
                        thinking += '</think>'
                        warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
                else:
                    thinking = ''
                    prediction = s
                return (prediction, thinking)
            split_func = model.split_thinking if hasattr(model, 'split_thinking') else split_thinking
            print(f'Prediction format: {os.getenv("SPLIT_THINK")},splitting func: {split_func}')
            tups = [split_func(x) for x in prediction]
            data['prediction'] = [x[0] for x in tups]
            data['thinking'] = [x[1] for x in tups]
        else:
            # data['prediction'] = [str(data_all[x]) for x in data['index']]
            # Add for agent evaluation
            if all(_is_structured_record(data_all[x]) for x in data['index']):
                data['prediction'] = [data_all[x]['prediction'] for x in data['index']]
                data['extra_records'] = [data_all[x]['extra_records'] for x in data['index']]
            else:
                data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
