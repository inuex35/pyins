# Copyright 2024 inuex35
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SP3 and CLK file downloader using HTTPS (inspired by gnss_lib_py)"""

import gzip
import os
import ssl
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import certifi


def gps_week_day(date: datetime) -> tuple[int, int]:
    """
    Calculate GPS week and day of week from date

    Parameters
    ----------
    date : datetime
        Date to convert

    Returns
    -------
    tuple
        (gps_week, day_of_week) where day_of_week is 0=Sunday
    """
    gps_epoch = datetime(1980, 1, 6)
    delta = date - gps_epoch
    gps_week = delta.days // 7
    gps_dow = delta.days % 7
    return gps_week, gps_dow


def download_sp3_cddis(date: datetime, product: str = "igs",
                      cache_dir: str = "./sp3_cache",
                      overwrite: bool = False) -> Optional[str]:
    """
    Download SP3 file from NASA CDDIS using HTTPS

    Parameters
    ----------
    date : datetime
        Date for which to download SP3 file
    product : str
        Product type: 'igs' (final), 'igr' (rapid), 'igu' (ultra-rapid),
        'cod' (CODE final), 'gfz' (GFZ rapid)
    cache_dir : str
        Directory to cache downloaded SP3 files
    overwrite : bool
        Whether to overwrite existing file

    Returns
    -------
    str or None
        Path to downloaded file or None if failed
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    gps_week, gps_dow = gps_week_day(date)
    year = date.year
    doy = date.timetuple().tm_yday  # Day of year

    # Construct filename based on product type
    if product == "igs":
        # IGS final product
        filename = f"igs{gps_week:04d}{gps_dow:01d}.sp3"
        url_path = f"/gnss/products/{gps_week:04d}/{filename}.Z"
    elif product == "igr":
        # IGS rapid product
        filename = f"igr{gps_week:04d}{gps_dow:01d}.sp3"
        url_path = f"/gnss/products/{gps_week:04d}/{filename}.Z"
    elif product == "igu":
        # IGS ultra-rapid (6-hour files)
        hour = (date.hour // 6) * 6
        filename = f"igu{gps_week:04d}{gps_dow:01d}_{hour:02d}.sp3"
        url_path = f"/gnss/products/{gps_week:04d}/{filename}.Z"
    elif product == "cod":
        # CODE final MGEX product (COD0MGXFIN)
        filename = f"COD0MGXFIN_{year:04d}{doy:03d}0000_01D_05M_ORB.SP3"
        url_path = f"/gnss/products/mgex/{gps_week:04d}/{filename}.gz"
    elif product == "gfz":
        # GFZ rapid MGEX product (GFZ0MGXRAP)
        filename = f"GFZ0MGXRAP_{year:04d}{doy:03d}0000_01D_05M_ORB.SP3"
        url_path = f"/gnss/products/mgex/{gps_week:04d}/{filename}.gz"
    else:
        raise ValueError(f"Unknown product type: {product}")

    # Check if file already exists
    local_path = cache_path / filename
    if local_path.exists() and not overwrite:
        print(f"SP3 file already exists: {local_path}")
        return str(local_path)

    # Try multiple servers
    servers = [
        ("https://igs.bkg.bund.de", url_path.replace("/gnss", "")),  # BKG server
        ("https://cddis.nasa.gov", url_path),  # NASA CDDIS
    ]

    # Create SSL context with certifi certificates
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    # Download compressed file
    compressed_path = local_path.with_suffix(local_path.suffix + ('.gz' if '.gz' in url_path else '.Z'))

    for base_url, path in servers:
        full_url = base_url + path
        print(f"Trying to download SP3 from: {full_url}")

        try:
            # Create request with headers
            request = urllib.request.Request(full_url)
            request.add_header('User-Agent', 'pyins SP3 downloader')

            # Download file
            with urllib.request.urlopen(request, context=ssl_context, timeout=30) as response:
                with open(compressed_path, 'wb') as f:
                    f.write(response.read())

            print(f"Downloaded compressed file: {compressed_path}")
            break  # Success, exit the loop

        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code} from {base_url}: {e.reason}")
            if e.code == 404:
                print(f"File not found on {base_url}")
            continue
        except urllib.error.URLError as e:
            print(f"URL Error from {base_url}: {e.reason}")
            continue
        except Exception as e:
            print(f"Error downloading from {base_url}: {e}")
            continue
    else:
        # All servers failed
        print("Failed to download from all servers")
        if compressed_path.exists():
            os.remove(compressed_path)
        return None

    # Decompress file
    try:
        if compressed_path.suffix == '.gz':
            # Handle gzip compression
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(local_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(compressed_path)
        elif compressed_path.suffix == '.Z':
            # Handle Unix compress (.Z files)
            # Try using uncompress command
            import subprocess
            try:
                subprocess.run(['uncompress', str(compressed_path)], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If uncompress not available, try gzip -d
                try:
                    subprocess.run(['gzip', '-d', str(compressed_path)], check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("Cannot decompress .Z file - uncompress/gzip not available")
                    os.remove(compressed_path)
                    return None

        if local_path.exists():
            print(f"Successfully downloaded and decompressed: {local_path}")
            return str(local_path)
        else:
            print("Decompression failed")
            return None

    except Exception as e:
        print(f"Error during decompression: {e}")
        if compressed_path.exists():
            os.remove(compressed_path)
        return None


def download_clk_cddis(date: datetime, product: str = "igs",
                       cache_dir: str = "./sp3_cache",
                       overwrite: bool = False) -> Optional[str]:
    """
    Download CLK file from NASA CDDIS using HTTPS

    Parameters
    ----------
    date : datetime
        Date for which to download CLK file
    product : str
        Product type: 'igs' (final), 'igr' (rapid), 'cod' (CODE final),
        'gfz' (GFZ rapid)
    cache_dir : str
        Directory to cache downloaded CLK files
    overwrite : bool
        Whether to overwrite existing file

    Returns
    -------
    str or None
        Path to downloaded file or None if failed
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    gps_week, gps_dow = gps_week_day(date)
    year = date.year
    doy = date.timetuple().tm_yday

    # Construct filename based on product type
    if product == "igs":
        # IGS final product
        filename = f"igs{gps_week:04d}{gps_dow:01d}.clk"
        url_path = f"/gnss/products/{gps_week:04d}/{filename}.Z"
    elif product == "igr":
        # IGS rapid product
        filename = f"igr{gps_week:04d}{gps_dow:01d}.clk"
        url_path = f"/gnss/products/{gps_week:04d}/{filename}.Z"
    elif product == "cod":
        # CODE final MGEX product
        filename = f"COD0MGXFIN_{year:04d}{doy:03d}0000_01D_30S_CLK.CLK"
        url_path = f"/gnss/products/mgex/{gps_week:04d}/{filename}.gz"
    elif product == "gfz":
        # GFZ rapid MGEX product
        filename = f"GFZ0MGXRAP_{year:04d}{doy:03d}0000_01D_30S_CLK.CLK"
        url_path = f"/gnss/products/mgex/{gps_week:04d}/{filename}.gz"
    else:
        raise ValueError(f"Unknown product type: {product}")

    # Check if file already exists
    local_path = cache_path / filename
    if local_path.exists() and not overwrite:
        print(f"CLK file already exists: {local_path}")
        return str(local_path)

    # Try multiple servers
    servers = [
        ("https://igs.bkg.bund.de", url_path.replace("/gnss", "")),  # BKG server
        ("https://cddis.nasa.gov", url_path),  # NASA CDDIS
    ]

    # Create SSL context with certifi certificates
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    # Download compressed file
    compressed_path = local_path.with_suffix(local_path.suffix + ('.gz' if '.gz' in url_path else '.Z'))

    for base_url, path in servers:
        full_url = base_url + path
        print(f"Trying to download CLK from: {full_url}")

        try:
            # Create request with headers
            request = urllib.request.Request(full_url)
            request.add_header('User-Agent', 'pyins CLK downloader')

            # Download file
            with urllib.request.urlopen(request, context=ssl_context, timeout=30) as response:
                with open(compressed_path, 'wb') as f:
                    f.write(response.read())

            print(f"Downloaded compressed file: {compressed_path}")
            break  # Success, exit the loop

        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code} from {base_url}: {e.reason}")
            continue
        except urllib.error.URLError as e:
            print(f"URL Error from {base_url}: {e.reason}")
            continue
        except Exception as e:
            print(f"Error downloading from {base_url}: {e}")
            continue
    else:
        # All servers failed
        print("Failed to download from all servers")
        if compressed_path.exists():
            os.remove(compressed_path)
        return None

    # Decompress file
    try:
        if compressed_path.suffix == '.gz':
            # Handle gzip compression
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(local_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(compressed_path)
        elif compressed_path.suffix == '.Z':
            # Handle Unix compress
            import subprocess
            try:
                subprocess.run(['uncompress', str(compressed_path)], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    subprocess.run(['gzip', '-d', str(compressed_path)], check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("Cannot decompress .Z file")
                    os.remove(compressed_path)
                    return None

        if local_path.exists():
            print(f"Successfully downloaded and decompressed: {local_path}")
            return str(local_path)
        else:
            print("Decompression failed")
            return None

    except Exception as e:
        print(f"Error during decompression: {e}")
        if compressed_path.exists():
            os.remove(compressed_path)
        return None


def get_best_sp3_product(date: datetime, cache_dir: str = "./sp3_cache") -> Optional[str]:
    """
    Download the best available SP3 product for a given date

    Strategy (following gnss_lib_py approach):
    - Within 2 days: Try IGU (ultra-rapid)
    - Within 3-17 days: Try IGR (rapid)
    - Older than 17 days: Try IGS (final)
    - Fallback: Try CODE or GFZ MGEX products

    Parameters
    ----------
    date : datetime
        Date for which to download SP3
    cache_dir : str
        Cache directory

    Returns
    -------
    str or None
        Path to downloaded SP3 file
    """
    now = datetime.now()
    age_days = (now - date).days

    # Determine product priority based on age
    if age_days <= 2:
        products = ['igu', 'igr', 'gfz', 'cod']
    elif age_days <= 17:
        products = ['igr', 'igs', 'gfz', 'cod']
    else:
        products = ['igs', 'cod', 'gfz', 'igr']

    # Try each product in order
    for product in products:
        print(f"Trying {product.upper()} product...")
        sp3_file = download_sp3_cddis(date, product, cache_dir)
        if sp3_file:
            return sp3_file

    return None
