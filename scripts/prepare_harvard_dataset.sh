name=Harvard
mkdir -p data/datasets/${name}
cd data/datasets/${name}

# Download real testing dataset
echo "Downloading ${name} dataset."
wget http://vision.seas.harvard.edu/qsfs/PSData.zip
unzip PSData.zip

cd PSData
ls cat/Objects/ > names.txt # prepare image list

# Back to root directory
cd ../../../../
cp scripts/${name}_objects.txt data/datasets/${name}/PSData/objects.txt
