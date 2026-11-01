import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset

def get_cifar100_extractor(root="./data"):
    """Khởi tạo Backbone ResNet18 và tải CIFAR-100 dataset"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Sử dụng thiết bị: {device}")
    
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()  # Loại bỏ lớp phân loại cuối, giữ vector 512 chiều
    backbone = backbone.to(device)
    backbone.eval()
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    
    return backbone, train_dataset, test_dataset, device

def extract_features(dataset, target_classes, backbone, device, batch_size=128):
    """Lọc ảnh theo lớp và trích xuất qua ResNet18"""
    indices = [i for i, label in enumerate(dataset.targets) if label in target_classes]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    features_list, labels_list = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feats = backbone(imgs)
            features_list.append(feats.cpu())
            labels_list.append(lbls)
            
    return torch.cat(features_list), torch.cat(labels_list)