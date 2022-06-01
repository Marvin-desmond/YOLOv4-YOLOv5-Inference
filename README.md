### Hi there! :wave:
***
### YOLOv5 :rocket: YOLOv4 ![darknet](https://camo.githubusercontent.com/6b3c6c1109586f5f3ddf8967fa4eaf787c7b45fe3df6d89111d6f9c7c1045769/687474703a2f2f706a7265646469652e636f6d2f6d656469612f66696c65732f6461726b6e65742d626c61636b2d736d616c6c2e706e67 =x30)

> This repo contains the inference pipelines for both AlexeyAB YOLO Darknet model and Ultralytics YOLO PyTorch model.
__*Concepts that have been illustrated include:*__
+ Loading the models and the configurations
+ Preprocessing pipelines for the models
+ Inference of the images
+ Post-processing
    - Non max suppression
    - Bounding box coordinates conversion
 + Showing detections on image
 + Visualize image

:pushpin: __NB: *During PyTorch serialization, the pickling tends to reference the Layer modules during torch.load, why so?*__
`torch.save`  basically only calls  `torch._save`
Inside this function there is a function named  `persistent_id`  defined and beside other things the return values of this function are pickled.

For  `torch.nn.Module`  this function does the following:
```
if  isinstance(obj, type) and  issubclass(obj, nn.Module): 
	if obj in serialized_container_types: 
		return  None 
	serialized_container_types[obj] = True 
	source_file = source = None  
	try: 
		source_file = inspect.getsourcefile(obj) 
		source = inspect.getsource(obj) 
	except Exception: 
		# saving the source is optional, 
		# so we can ignore any errors 
		warnings.warn("Couldn't retrieve source code for container of "  
		"type " + obj.__name__ + ". It won't be checked "  
		"for correctness upon loading.") 
	return ('module', obj, source_file, source)
```
Which means the source code and the source file are pickled to. This results in the fact, that your source file will be parsed again during loading (and compared to the pickled source code to generate warnings if necessary). This source file is then used for model creation if the changes can be merged automatically. And thus adding new methods is valid as long as you don’t change the existing ones in a way that prevents python from merging automatically.
>![PyTorch Forums](https://discuss.pytorch.org/uploads/default/original/2X/3/38d28fd067a1a8f263e14507942b2e38e49b771a.png =18x18)     [Question about serialization while saving models in PyTorch](https://discuss.pytorch.org/t/question-about-serialization-while-saving-models-in-pytorch/23212)

__*Hence, the files `yolo.py` and `common.py` cannot be deleted since serialization uses them during `torch.load`*__

[[https://discuss.pytorch.org/uploads/default/original/2X/3/38d28fd067a1a8f263e14507942b2e38e49b771a.png|width=400px]]

[![linkedin](https://img.shields.io/badge/LinkedIn-166FC5?style=border-radius:3px&logo=LinkedIn&logoColor=white)
](https://www.linkedin.com/in/marvin-mboya-b7bb81195/) [![github](https://img.shields.io/badge/GitHub-000000?style=border-radius:3px&logo=GitHub&logoColor=white)](https://github.com/Marvin-desmond)
