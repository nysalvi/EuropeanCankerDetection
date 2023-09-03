from typing import Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
import torch.utils.data as data_utils
from dataclasses import asdict
import pandas as pd
import torch
import os

class File():
    def __init_(self, path):
        self.path = path
    #make_dataset override
    def make_dataset(self,
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)
        if class_to_idx is None:
            _, class_to_idx = find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None    
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):        
            class_index = class_to_idx[target_class] 
            #NEW LINE : Class and index folder structure gets united again; 
            # e.g '0', 'Negativo' becomes : '0 - negativo'
            correct_path = str(class_index) + ' - ' + target_class      
            target_dir = os.path.join(directory, correct_path)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances
    #find_classes override    
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        labels = []        
        idx_label = {}
        for root, dirs, files in os.walk(directory):
            if dirs == []:            
                str_split = root.split(f'{os.sep}')[-1]
                pair = str_split.split(' - ') if len(str_split.split(' - ')) == 2 else False
                if pair:
                    num, label = pair     
                    if not label in labels:
                        labels.append(label)            
                    idx_label.update({label : int(num)})
        return (labels, idx_label)
        
    def load_dataset(self, transforms:Compose, batch_size) -> data_utils.DataLoader:        
        image_folder = ImageFolder(self.path, transform=transforms)
        return data_utils.DataLoader(image_folder, batch_size=batch_size, shuffle=True)
    
    def save_checkpoint(self, data:pd.DataFrame, **kargs):
        for x, y in kargs.items(): torch.save({x:y}, f'{self.path}{os.sep}{x}.pt')
        data_path = f'{self.path}{os.sep}data.csv'
        if os.path.exists(data_path): data.to_csv(data_path, index=False, header=False, mode='a')
        else: data.to_csv(data_path, index=False, header=True, mode='w')

    def load_checkpoint(self):        
        files = os.listdir()                    
        return {torch.load(f'{self.path}{os.sep}{f}') for f in files if f.endswith('.pt')}

    #def save_metrics():
    #os.makedirs(Info.PATH, exist_ok=True)
    #os.makedirs(Info.BoardX, exist_ok=True)

    def clear_progress(self):
        if os.path.exists(self.path):
            os.rmdir(self.path)

ImageFolder.find_classes = File.find_classes
ImageFolder.make_dataset = File.make_dataset