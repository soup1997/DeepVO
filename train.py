
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from tqdm import tqdm
from utils.dataloader import *
from utils.helpers import *
from model.deepvo import DeepVO

import warnings
warnings.filterwarnings('ignore')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def valid_one_epoch(valid_loader):
    model.eval()
    valid_loss = 0.0
    position_error = 0.0
    rotation_error = 0.0
 
    progress_bar = tqdm(valid_loader, total=len(valid_loader), desc=f'Epoch {epoch}/{num_epochs}, Valid Loss: 0.0000')
    
    with torch.no_grad():
        for batch_idx, (img, gt) in enumerate(progress_bar):
            images, pose = img.to(device), gt.to(device)
            
            output = model(images)
            loss = model.criterion(output, pose)
            valid_r_error, valid_p_error = model.calc_err(output, pose) 

            valid_loss += loss.item()
            rotation_error += valid_r_error.item()
            position_error += valid_p_error.item()

            progress_bar.set_description(f'Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss / (batch_idx + 1):.10f}, Valid position error: {position_error / (batch_idx + 1):.10f}, Valid rotation error: {rotation_error / (batch_idx + 1):.10f}')
    print("\n")

    valid_loss /= len(valid_loader)
    position_error /= len(valid_loader)
    rotation_error /= len(valid_loader)
    progress_bar.close()

    return valid_loss, position_error, rotation_error

def train_one_epoch(epoch, train_loader):
    model.train()

    train_loss = 0.0
    position_error = 0.0
    rotation_error = 0.0

    progress_bar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}, Train Loss: 0.0000')

    for batch_idx, (img, gt) in enumerate(progress_bar):
        optimizer.zero_grad()
    
        img, pose = img.to(device), gt.to(device)
        output = model(img)  # output is (roll, pitch, yaw, x, y, z)
        loss = model.criterion(output, pose)

        train_r_error, train_p_error = model.calc_err(output, pose)

        train_loss += loss.item()
        rotation_error += train_r_error.item()
        position_error += train_p_error.item()

        loss.backward()
        
        optimizer.step()
        progress_bar.set_description(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss / (batch_idx + 1):.10f}, Train position error: {position_error / (batch_idx + 1):.10f}, Train rotation error: {rotation_error / (batch_idx + 1):.10f}')

    train_loss /= len(train_loader)
    position_error /= len(train_loader)
    progress_bar.close()

    return train_loss, position_error, rotation_error


if __name__=='__main__':
    torch.cuda.empty_cache()

    args = arg_parse()
    config = parse_config_yaml(args.config_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    dataset_dir = config['dataset_path']
    save_path = config['model_save_path']

    # Create the model
    model = DeepVO(batchNorm=True)
    model.load_pretrained_weight()
    summary(model, (config['HYPER_PARAMS']['batch_size'], config['MODEL']['lstm_seq_len'], 3, config['MODEL']['height'], config['MODEL']['width']))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config['HYPER_PARAMS']['lr'], 
                                 weight_decay=config['HYPER_PARAMS']['weight_decay'])
    
    scheduler = ExponentialLR(optimizer, 
                              gamma=config['HYPER_PARAMS']['lr_decay_factor'])
    
    num_epochs = config['HYPER_PARAMS']['epochs']
    
    torch.set_printoptions(sci_mode=False, precision=10)
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(1, num_epochs + 1):
        train_loader, valid_loader, test_loader = load_dataset(dataset_dir, 
                                                               batch_size=config['HYPER_PARAMS']['batch_size'], 
                                                               seq_len=config['MODEL']['lstm_seq_len'], 
                                                               width=config['MODEL']['width'], 
                                                               height=config['MODEL']['height'], 
                                                               shuffle=True)
        
        train_loss, train_t_err, train_q_err = train_one_epoch(epoch, train_loader)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Translation Error/Train", train_t_err, epoch)
        writer.add_scalar("Orientation Error/Train", train_q_err, epoch)


        valid_loss, valid_t_err, valid_q_err = valid_one_epoch(valid_loader)
        writer.add_scalar("Loss/Valid", valid_loss, epoch)
        writer.add_scalar("Translation Error/Valid", valid_t_err, epoch)
        writer.add_scalar("Orientation Error/Valid", valid_q_err, epoch)
        
        if epoch % config['HYPER_PARAMS']['save_step'] == 0:
            model.eval()
            torch.save(model.state_dict(), save_path + f"Epoch:{epoch} | Position Error:{valid_t_err:.6f} | Rotation Error: {valid_q_err:.6f}.pth")
        
        if epoch % config['HYPER_PARAMS']['lr_step'] == 0:
            writer.add_scalar("Learning rate", get_lr(optimizer), epoch)
            scheduler.step()

    writer.close()