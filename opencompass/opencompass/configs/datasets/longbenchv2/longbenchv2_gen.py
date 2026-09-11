from mmengine.config import read_base

with read_base():
    from .longbenchv2_0shot_gen import LongBenchv2_0shot_datasets
    from .longbenchv2_cot_gen import LongBenchv2_cot_datasets

LongBenchv2_datasets = LongBenchv2_0shot_datasets + LongBenchv2_cot_datasets
