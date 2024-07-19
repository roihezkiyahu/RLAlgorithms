import yaml
import os
from google.cloud import storage


def modify_config_and_upload(config_path, gcs_bucket_name):
    gcs_config_path = f"configs/{os.path.basename(config_path)}"
    # Read the YAML configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Modify the folder path to point to GCS
    new_folder_path = f'gs://{gcs_bucket_name}/logging/trainer_config_Breakout_llp5'
    config['trainer']['folder'] = new_folder_path

    # Save the modified configuration to a new file
    modified_config_path = 'modified_' + os.path.basename(config_path)
    with open(modified_config_path, 'w') as file:
        yaml.dump(config, file)

    # Upload the modified configuration to GCS
    upload_to_gcs(gcs_bucket_name, modified_config_path, gcs_config_path)
    print(f"Modified config uploaded to gs://{gcs_bucket_name}/{gcs_config_path} v1")
    new_config_path = f"gs://{gcs_bucket_name}/{gcs_config_path}"
    return new_config_path


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    print("GOOGLE_APPLICATION_CREDENTIALS:", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))


    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


# Example usage
if __name__ == "__main__":
    config_path = os.path.join("..", "modeling", "configs", "trainer_config_Breakout_llp5.yaml")
    gcs_bucket_name = 'rl-on-gcp-427712-rl-algos'

    modify_config_and_upload(config_path, gcs_bucket_name)
