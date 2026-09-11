# flake8: noqa
# yapf: disable
import json
import os
import os.path as osp
from datetime import datetime
from typing import List

from .default import DefaultSummarizer


class ScibenchSummarizer(DefaultSummarizer):
    """Scibench Summarizer that merges subset results into a single file.
    
    This summarizer extends the default summarizer to automatically merge
    all scibench subset results into a consolidated file after evaluation.
    
    Args:
        scibench_subsets (list[str]): List of scibench subset names in the desired order.
            Example: ['atkins', 'calculus', 'chemmc', 'class', 'diff', 'fund', 
                      'matter', 'quan', 'stat', 'thermo']
    """
    
    def __init__(self, scibench_subsets: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.scibench_subsets = scibench_subsets or [
            'atkins', 'calculus', 'chemmc', 'class', 'diff',
            'fund', 'matter', 'quan', 'stat', 'thermo'
        ]
        
    def summarize(
        self,
        output_path: str = None,
        time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        
        # Call parent summarize first
        super().summarize(output_path, time_str)
        
        # Then merge scibench results
        self._merge_scibench_results()
    
    def _merge_scibench_results(self):
        """Merge all scibench subset results into a consolidated file."""
        results_dir = osp.join(self.work_dir, 'results')
        
        if not osp.exists(results_dir):
            self.logger.warning(f"Results directory not found: {results_dir}")
            return
        
        self.logger.info("Starting Scibench results merging...")
        
        # Iterate through each model's results directory
        processed_models = 0
        for model_dir in os.listdir(results_dir):
            model_path = osp.join(results_dir, model_dir)
            if not osp.isdir(model_path):
                continue
            
            self.logger.info(f"Merging results for model: {model_dir}")
            
            # Merge results according to scibench_subsets order
            merged_data = {
                'model': model_dir,
                'subsets': [],
                'total_accuracy': 0.0,
                'total_questions': 0,
                'correct_questions': 0,
                'details': {}
            }
            
            total_correct = 0
            total_questions = 0
            question_offset = 0  # For renumbering question IDs
            
            for subset_name in self.scibench_subsets:
                subset_file = osp.join(model_path, f'scibench-{subset_name}.json')
                
                if not osp.exists(subset_file):
                    self.logger.debug(f"Subset file not found: scibench-{subset_name}.json")
                    continue
                
                # Read subset results
                try:
                    with open(subset_file, 'r', encoding='utf-8') as f:
                        subset_data = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to read {subset_file}: {e}")
                    continue
                
                # Record subset information
                subset_details = subset_data.get('details', {})
                num_questions = len([k for k in subset_details if k != 'type'])
                
                subset_info = {
                    'name': f'scibench-{subset_name}',
                    'accuracy': subset_data.get('accuracy', 0.0),
                    'num_questions': num_questions,
                    'question_range': [question_offset, question_offset + num_questions - 1] if num_questions > 0 else []
                }
                merged_data['subsets'].append(subset_info)
                
                # Calculate total accuracy
                if num_questions > 0:
                    # Calculate correct answers for this subset
                    correct = int(subset_data.get('accuracy', 0.0) * num_questions / 100.0 + 0.5)
                    total_correct += correct
                    total_questions += num_questions
                    
                    self.logger.debug(f"  {subset_name:10s}: {num_questions:2d} questions, "
                                    f"accuracy {subset_data.get('accuracy', 0.0):6.2f}%")
                
                # Merge detailed information with renumbering
                if 'type' in subset_details:
                    merged_data['details']['type'] = subset_details['type']
                
                for key, value in subset_details.items():
                    if key == 'type':
                        continue
                    # Renumber question IDs
                    try:
                        question_id = int(key)
                        new_id = str(question_offset + question_id)
                        merged_data['details'][new_id] = value.copy() if isinstance(value, dict) else value
                        # Add subset information tag
                        if isinstance(merged_data['details'][new_id], dict):
                            merged_data['details'][new_id]['subset'] = f'scibench-{subset_name}'
                            merged_data['details'][new_id]['original_id'] = key
                    except (ValueError, TypeError):
                        # If key is not a number, keep it as is
                        if key not in merged_data['details']:
                            merged_data['details'][key] = value
                
                question_offset += num_questions
            
            # Calculate overall accuracy
            if total_questions > 0:
                merged_data['total_accuracy'] = (total_correct / total_questions) * 100.0
                merged_data['total_questions'] = total_questions
                merged_data['correct_questions'] = total_correct
            
            # Save merged results
            merged_file = osp.join(model_path, 'scibench.json')
            try:
                with open(merged_file, 'w', encoding='utf-8') as f:
                    json.dump(merged_data, f, indent=4, ensure_ascii=False)
                
                self.logger.info(f"✓ Merged results saved: {merged_file}")
                self.logger.info(f"  Total questions: {total_questions}, "
                               f"Correct: {total_correct}, "
                               f"Accuracy: {merged_data['total_accuracy']:.2f}%")
                
                processed_models += 1
            except Exception as e:
                self.logger.error(f"Failed to save merged file: {e}")
        
        if processed_models == 0:
            self.logger.warning("No model results found for merging")
        else:
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Scibench merging completed! Processed {processed_models} model(s)")
            self.logger.info(f"{'='*60}")
