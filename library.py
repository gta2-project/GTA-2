"""Launch the separate GTA-2 Prepare / Library / Run web application."""
import os
from pathlib import Path

from dotenv import load_dotenv
import uvicorn


def main():
    root = Path(__file__).resolve().parent
    load_dotenv(root / '.env')
    host = os.getenv('GTA_LIBRARY_HOST', '127.0.0.1')
    port = int(os.getenv('GTA_LIBRARY_PORT', '8001'))
    print(f'GTA-2 library app: http://{host}:{port}/admin')
    print('Pages: /admin, /library, /run (main.py remains separate)')
    uvicorn.run('api.library_app:app', host=host, port=port)


if __name__ == '__main__':
    main()
