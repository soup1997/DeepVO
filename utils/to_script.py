import torch
from model.deepvo import DeepVO

if __name__=='__main__':
    traine_model_path = '/home/smeet/catkin_ws/src/Visual-Inertial-Odometry/DeepVO/trained/'
    model = DeepVO(batchNorm=True)
    model.load_state_dict(torch.load(traine_model_path + 'DeepVO.pth'))
    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.save(traine_model_path + 'DeepVO_scripted.pt')