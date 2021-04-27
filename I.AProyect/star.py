from convertA import process_image_dataset

# Set square dimensions of images
size = (32,32) # 32 by 32 pixels

# Set number of batches
batch = 1

# Source of image dataset (Use absolute path)
source = '/dataset'

# Destination of processed dataset (use absolute path)
destination = '/dta'

# Process dataset
process_image_dataset(source, destination, size, batch)