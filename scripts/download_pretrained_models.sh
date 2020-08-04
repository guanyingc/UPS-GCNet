path="data/models_ECCV2020/"
mkdir -p $path
cd $path

# Download pre-trained model
for model in "GCNet.pth" "PS-FCN_B_S_32.pth" "LCNet_CVPR2019.pth"; do
    wget http://www.visionlab.cs.hku.hk/data/UPS-GCNet/models_release/${model}
done
