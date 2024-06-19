# -*- coding: utf-8 -*-
"""
Script to download an entire folder from Google Drive.

This script allows users to download a specified folder and its contents from Google Drive
to a local path using the Google Drive API. The script supports multiprocessing to speed up
the download process.

Usage:
    Save this script as download_p.py and run the following command in the terminal:
    python download_p.py --folder_id YOUR_FOLDER_ID --save_path /path/to/save --service_account_file /path/to/service_account.json --workers 4

Arguments:
    --folder_id: The ID of the Google Drive folder to download.
    --save_path: The local path to save the downloaded folder.
    --service_account_file: The path to the service account credentials JSON file.
    --workers: The number of processes to use, default is 4.

Example:
    python download_p.py --folder_id 1a2B3cD4eFgH5I6jK7L8mN9oP --save_path /home/user/downloads --service_account_file /home/user/service_account.json --workers 4
"""

import os
import io
import pathlib
import json
import logging
import time
import argparse
from multiprocessing import Pool, Manager, Value, Lock
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from googleapiclient.errors import HttpError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_progress(progress_file):
    """Load the download progress from a file.

    Args:
        progress_file (str): Path to the file storing download progress.

    Returns:
        set: A set of downloaded files.
    """
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logger.error("Progress file is empty or corrupted, starting fresh.")
            return set()
    return set()

def save_progress(progress_file, downloaded_files):
    """Save the download progress to a file.

    Args:
        progress_file (str): Path to the file storing download progress.
        downloaded_files (list): List of downloaded files.
    """
    with open(progress_file, 'w') as f:
        json.dump(list(downloaded_files), f)

def get_total_size(service, folder_id, lock, total_size):
    """Calculate the total size of the folder.

    Args:
        service (Resource): Google Drive API service instance.
        folder_id (str): ID of the folder.
        lock (Lock): Lock for synchronization.
        total_size (Value): Shared variable for total size.
    """
    page_token = None
    while True:
        query = f"'{folder_id}' in parents and trashed = false"
        try:
            results = service.files().list(q=query, fields="nextPageToken, files(id, name, mimeType, size)", pageToken=page_token).execute()
        except HttpError as error:
            logger.error(f"An error occurred: {error}")
            return
        
        items = results.get('files', [])
        for item in items:
            mime_type = item['mimeType']
            if mime_type == 'application/vnd.google-apps.folder':
                get_total_size(service, item['id'], lock, total_size)
            else:
                if 'size' in item:
                    with lock:
                        total_size.value += int(item['size'])
        page_token = results.get('nextPageToken')
        if not page_token:
            break

def download_file(args):
    """Download a single file.

    Args:
        args (tuple): Arguments needed for downloading a file.
    """
    service, file_id, file_name, save_path, retries, lock, downloaded_files, total_size, downloaded_size, progress_file = args
    if file_name in downloaded_files:
        logger.info(f"Skipping {file_name}, already downloaded.")
        return
    
    for attempt in range(retries):
        try:
            request = service.files().get_media(fileId=file_id)
            fh = io.FileIO(os.path.join(save_path, file_name), 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            previous_progress = 0
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    current_progress = status.resumable_progress
                    with lock:
                        downloaded_size.value += (current_progress - previous_progress)
                        previous_progress = current_progress
                        overall_progress = int((downloaded_size.value / total_size.value) * 100)
                    logger.info(f"Download {file_name} {int(status.progress() * 100)}% (Overall: {overall_progress}%)")
            fh.close()
            with lock:
                downloaded_files.append(file_name)
                save_progress(progress_file, downloaded_files)
            break
        except HttpError as error:
            logger.error(f"Error downloading {file_name}: {error}")
            if attempt < retries - 1:
                logger.info(f"Retrying {file_name} ({attempt + 1}/{retries})...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to download {file_name} after {retries} attempts.")
        finally:
            if 'fh' in locals() and not fh.closed:
                fh.close()

def download_folder(service, folder_id, save_path, lock, downloaded_files, total_size, downloaded_size, progress_file, workers):
    """Download all files in the folder.

    Args:
        service (Resource): Google Drive API service instance.
        folder_id (str): ID of the folder.
        save_path (str): Local path to save the files.
        lock (Lock): Lock for synchronization.
        downloaded_files (list): List of downloaded files.
        total_size (Value): Shared variable for total size.
        downloaded_size (Value): Shared variable for downloaded size.
        progress_file (str): Path to the file storing download progress.
        workers (int): Number of processes to use.
    """
    # Ensure save path exists
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    tasks = []
    page_token = None
    while True:
        query = f"'{folder_id}' in parents and trashed = false"
        results = None
        for attempt in range(3):  # Retry mechanism
            try:
                results = service.files().list(q=query, fields="nextPageToken, files(id, name, mimeType)", pageToken=page_token).execute()
                break
            except HttpError as error:
                logger.error(f"Error listing folder {folder_id}: {error}")
                if attempt < 2:
                    logger.info(f"Retrying listing folder {folder_id} ({attempt + 1}/3)...")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to list folder {folder_id} after 3 attempts.")
                    return

        if not results:
            return

        items = results.get('files', [])
        for item in items:
            file_id = item['id']
            file_name = item['name']
            mime_type = item['mimeType']
            if mime_type == 'application/vnd.google-apps.folder':
                logger.info(f"Entering folder {file_name}")
                download_folder(service, file_id, os.path.join(save_path, file_name), lock, downloaded_files, total_size, downloaded_size, progress_file, workers)
            else:
                tasks.append((service, file_id, file_name, save_path, 3, lock, downloaded_files, total_size, downloaded_size, progress_file))
        page_token = results.get('nextPageToken')
        if not page_token:
            break

    with Pool(processes=workers) as pool:
        pool.map(download_file, tasks)

def main(folder_id, save_path, service_account_file, workers):
    """Main function to set up the service and start downloading.

    Args:
        folder_id (str): The ID of the Google Drive folder to download.
        save_path (str): The local path to save the downloaded folder.
        service_account_file (str): The path to the service account credentials JSON file.
        workers (int): The number of processes to use.
    """
    # Set up service account credentials
    SCOPES = ['https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
    
    # Build the service
    service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
    
    progress_file = os.path.join(save_path, 'download_progress.json')
    manager = Manager()
    lock = manager.Lock()
    downloaded_files = manager.list(load_progress(progress_file))  # Use manager.list() instead of set
    total_size = manager.Value('i', 0)
    downloaded_size = manager.Value('i', 0)

    # Calculate total size
    logger.info("Calculating total size...")
    get_total_size(service, folder_id, lock, total_size)
    logger.info(f"Total size to download: {total_size.value} bytes")

    # Start downloading
    download_folder(service, folder_id, save_path, lock, downloaded_files, total_size, downloaded_size, progress_file, workers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Google Drive folder.')
    parser.add_argument('--folder_id', required=True, help='Google Drive folder ID to download')
    parser.add_argument('--save_path', required=True, help='Local path to save the downloaded folder')
    parser.add_argument('--service_account_file', required=True, help='Path to the service account credentials JSON file')
    parser.add_argument('--workers', default=4, type=int, help='The number of processes, default is 4')

    args = parser.parse_args()

    main(args.folder_id, args.save_path, args.service_account_file, args.workers)
