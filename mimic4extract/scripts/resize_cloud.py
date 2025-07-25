import time
from PIL import Image
from tqdm import tqdm
import os
import argparse
from multiprocessing.dummy import Pool as ThreadPool
from google.cloud import storage
import io
import shutil

def resize_images(blob):
    basewidth = 512
    filename = blob.name.split('/')[-1]
    
    # Download image from GCS
    img_bytes = blob.download_as_bytes()
    img = Image.open(io.BytesIO(img_bytes))

    # Resize image
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize))
    
    # Save resized image locally
    img.save(f'{args.cxr_path}/resized/{filename}')


def list_gcs_images(bucket_name, prefix):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    return list(bucket.list_blobs(prefix=prefix))


parser = argparse.ArgumentParser(description="Resize the CXR images from GCS bucket.")
parser.add_argument('mimic_cxr_bucket', type=str, help="GCS bucket name containing MIMIC-CXR images.")
parser.add_argument('cxr_path', type=str, default='data/cxr', help="Local directory where processed images should be stored.")
args, _ = parser.parse_known_args()

print('Starting processing of CXR images from GCS bucket.')

# Ensure output directory exists
if not os.path.exists(os.path.join(args.cxr_path, 'resized')):
    os.makedirs(os.path.join(args.cxr_path, 'resized'))

# List all images in the GCS bucket with the specified prefix
prefix = 'mimic-cxr-jpg/2.0.0/files/'
blobs_all = list_gcs_images(args.mimic_cxr_bucket, prefix)
print('Total images found:', len(blobs_all))

# Filter out already processed images
paths_done = [os.path.basename(path) for path in os.listdir(os.path.join(args.cxr_path, 'resized'))]
blobs = [blob for blob in blobs_all if os.path.basename(blob.name) not in paths_done]
print('Images left to process:', len(blobs))

# Set up multi-threaded processing
threads = 10
for i in tqdm(range(0, len(blobs), threads)):
    blobs_subset = blobs[i: i + threads]
    pool = ThreadPool(len(blobs_subset))
    pool.map(resize_images, blobs_subset)
    pool.close()
    pool.join()

# Copy metadata files from GCS bucket to local directory
storage_client = storage.Client()
bucket = storage_client.bucket(args.mimic_cxr_bucket)

metadata_files = ['mimic-cxr-2.0.0-metadata.csv', 'mimic-cxr-2.0.0-chexpert.csv']
for file in metadata_files:
    blob = bucket.blob(file)
    if blob.exists():
        blob.download_to_filename(os.path.join(args.cxr_path, file))
        print(f'Copied {file} to local directory.')
    else:
        print(f'There is no file: {file}')