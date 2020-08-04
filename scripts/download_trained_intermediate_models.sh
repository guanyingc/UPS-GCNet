path="data/intermediate_model/"
mkdir -p $path
cd $path

for model in "L-Net1.pth" "N-Net_from_scratch.pth" "N-Net.pth" "L1_N_L2-Net.pth"; do
    wget http://www.visionlab.cs.hku.hk/data/UPS-GCNet/models_release/intermediate_net/${model}
done

