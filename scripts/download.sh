# Create the coco directory and cd into it
mkdir coco
cd coco

# Create the images directory and cd into it
mkdir images
cd images

# Download the dataset zip files in parallel
wget -c http://images.cocodataset.org/zips/train2017.zip &
wget -c http://images.cocodataset.org/zips/val2017.zip &
wget -c http://images.cocodataset.org/zips/test2017.zip &
wget -c http://images.cocodataset.org/zips/unlabeled2017.zip &
wait

# Unzip the dataset zip files in parallel
unzip train2017.zip &
unzip val2017.zip &
unzip test2017.zip &
unzip unlabeled2017.zip &
wait

# Remove the zip files
rm train2017.zip
rm val2017.zip
rm test2017.zip
rm unlabeled2017.zip

# Go back to the coco directory
cd ../

# Download the annotation zip files in parallel
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip &
wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip &
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip &
wget -c http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip &
wait

# Unzip the annotation zip files in parallel
unzip annotations_trainval2017.zip &
unzip stuff_annotations_trainval2017.zip &
unzip image_info_test2017.zip &
unzip image_info_unlabeled2017.zip &
wait

# Remove the zip files
rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip
rm image_info_test2017.zip
rm image_info_unlabeled2017.zip
