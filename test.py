import torch
from model.deepvo import DeepVO
from utils.helpers import *
from utils.util import *
from utils.dataloader import *

if __name__=='__main__':
    seq = '05'
    choice = '6dof'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepVO(batchNorm=True)
    model.load_state_dict(torch.load('./trained/Epoch:230 | Position Error:0.064374 | Rotation Error: 0.001241.pth'))
    model.to(device)
    model.eval()

    test_loss = 0.0
    position_error = 0.0
    rotation_error = 0.0

    H = np.identity(4, dtype=np.float32)

    args = arg_parse()
    config = parse_config_yaml(args.config_path)
    dataset_dir = config['dataset_path']
    save_path = config['model_save_path']
    
    with torch.no_grad():
        if choice == '12poses':
            output_dir = f'/home/smeet/catkin_ws/src/Visual-Inertial-Odometry/DeepVO/output/output_{seq}.txt'

        elif choice == '6dof':
            output_dir = f'/home/smeet/catkin_ws/src/Visual-Inertial-Odometry/DeepVO/output/output_6dof_{seq}.txt'

        dataset = DataLoader(ImageSequenceDataset(dataset_dir=dataset_dir, kitti_sequence=seq, seq_len=2), batch_size=1, shuffle=False)
        
        with open(output_dir, 'w') as f:
            for batch_idx, (img, gt) in enumerate(dataset):
                print(f"{batch_idx + 1}/{len(dataset)}")
                img, local_pose = img.to(device), gt.to(device)
                output = model(img)
                loss = model.criterion(output, local_pose)
                test_r_error, test_p_error = model.calc_err(output, local_pose)

                test_loss += loss.item()
                rotation_error += test_r_error.item()
                position_error += test_p_error.item()

                batch_size, seq_len, _ = output.size()

                for b in range(batch_size):
                    for t in range(seq_len):
                        pose_output = output[b, t].cpu().numpy()

                        if choice == '12poses':
                            pose_output = pose_to_SE3(pose_output)
                            H = H @ pose_output
                            pose_output = H[:3, :4].reshape(-1)

                        pose_output_str = ' '.join(['{:.6f}'.format(val) for val in pose_output])
                        f.write(pose_output_str.strip() + "\n")

        test_loss /= len(dataset)
        position_error /= len(dataset)
        rotation_error /= len(dataset)
        print(test_loss, position_error, rotation_error)