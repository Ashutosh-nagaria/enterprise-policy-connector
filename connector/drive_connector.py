import os
import io
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pickle
import docx  # this reads .docx files if a file wasn't converted to a Google Doc

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
PARENT_FOLDER_ID = '1jcrx_r1P1ZTJHIFvaK08NujBUsyQthb5'

def authenticate():
    creds = None
    if os.path.exists('credentials/token.pickle'):
        with open('credentials/token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials/client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('credentials/token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return creds

def list_subfolders(service, parent_id):
    results = service.files().list(
        q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder'",
        fields="files(id, name)"
    ).execute()
    return results.get('files', [])

def list_files_in_folder(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType)"
    ).execute()
    return results.get('files', [])

def get_file_text(service, file_id, mime_type):
    """
    Pulls the actual text content of a file.
    If it's a native Google Doc, we 'export' it as plain text.
    If it's still a raw .docx file, we download the bytes and read them with python-docx.
    """
    if mime_type == 'application/vnd.google-apps.document':
        # Native Google Doc — export as plain text
        request = service.files().export_media(fileId=file_id, mimeType='text/plain')
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        return fh.getvalue().decode('utf-8')
    else:
        # Raw .docx file — download bytes, then extract text
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        document = docx.Document(fh)
        return '\n'.join(p.text for p in document.paragraphs)

def run():
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    zones = list_subfolders(service, PARENT_FOLDER_ID)
    print(f"Found {len(zones)} zone folders:")

    all_docs = []
    for zone in zones:
        print(f"\nZone: {zone['name']}")
        files = list_files_in_folder(service, zone['id'])
        for f in files:
            print(f"  Reading: {f['name']}...")
            text = get_file_text(service, f['id'], f['mimeType'])
            all_docs.append({
                'zone': zone['name'],
                'file_id': f['id'],
                'name': f['name'],
                'text': text,
            })
            # Print just the first 200 characters as a sanity check
            print(f"    Preview: {text[:200].strip()!r}")

    print(f"\nTotal documents pulled: {len(all_docs)}")
    return all_docs

if __name__ == '__main__':
    run()