import torch
import torch.onnx
import torchvision.models as models
# Set the CPU to be used to export the model.
device = torch.device("cpu") 
 
def convert():
# The model definition comes from the torchvision. The model file generated in the example is based on the ResNet-50 model.
    model = models.resnet50(pretrained = True)  
    ##model = models.alexnet(pretrained=True)
    resnet50_model = torch.load('runs/Ant_05-19-35-44/nn/last_Ant_ep_500_rew__6603.7_.pth', map_location='cpu')
    model.load_state_dict(resnet50_model) 
 
    batch_size = 1 # Size of the batch processing
    input_shape = (60,256, 128, 64,8)  #(3, 224, 224) Input data. Replace it with the actual shape.

    # Set the model to inference mode.
    model.eval()

    dummy_input = torch.randn(batch_size, *input_shape) # Define the input shape.
    torch.onnx.export(model, 
                      dummy_input, 
                      "resnet50_official.onnx", 
                      input_names = ["input"], # Construct the input name.
                      output_names = ["output"], # Construct the output name.
                      opset_version=11, # Currently, the ATC tool supports only opset_version=11.
                      dynamic_axes={"input":{0:"batch_size"}, "output":{0:"batch_size"}}) # Dynamic axes of the output is supported.
                       
     
if __name__ == "__main__":
    convert()
    #model = models.resnet50(pretrained = True)  
    #resnet50_model = torch.load('runs/Ant_05-19-35-44/nn/last_Ant_ep_500_rew__6603.7_.pth', map_location='cpu')

    #print(resnet50_model.shape)