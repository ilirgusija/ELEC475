# Set your remote server details
REMOTE_USER="ilir"
REMOTE_HOST="192.168.1.239"
REMOTE_DIR="/home/ilir/school/elec475"

# Function to train, tar, and scp
train_and_transfer() {
    # Run the training script with provided arguments
    python3 train_classifier.py -gamma $1 -e $2 -m $3 -lr $4 -d $5 -b $6 -s $7 -p $8 -d_p $9 -cuda ${10}

    # Create a unique identifier for the files
    ID=$(date +%s)

    # Create a tarball of the output files
    tar -czvf "../output_$3_$ID.tar.gz" -C ../ output

    # # Copy the tarball to the remote machine
    # scp "../output_$3_$ID.tar.gz" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

    # # Optionally, remove the tarball after transfer
    # rm "../output_$3_$ID.tar.gz"
}

# Call the function with different parameters
train_and_transfer 0.1 30 resnet_18 0.01 ../data/Kitti8_ROIs 512 ../output/classifier_resnet_18.pth ../output/loss_resnet_18.png neither Y
train_and_transfer 0.1 30 se_resneXt 0.001 ../data/Kitti8_ROIs 256 ../output/classifier_se_resneXt.pth ../output/loss_se_resneXt.png neither Y
train_and_transfer 0.1 30 resnet 0.01 ../data/Kitti8_ROIs 512 ../output/classifier_resnet.pth ../output/loss_resnet.png neither Y
train_and_transfer 0.1 30 se_resnet 0.0001 ../data/Kitti8_ROIs 256 ../output/classifier_se_resnet.pth ../output/loss_se_resnet.png neither Y
