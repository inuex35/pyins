#!/usr/bin/env python3
"""
Improved SP3 downloader using gnss_lib_py approach with FTP support
"""

import os
import gzip
import shutil
from ftplib import FTP, FTP_TLS
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import subprocess

def gps_week_day(date: datetime) -> Tuple[int, int]:
    """Calculate GPS week and day of week from date"""
    gps_epoch = datetime(1980, 1, 6)
    delta = date - gps_epoch
    gps_week = delta.days // 7
    gps_dow = delta.days % 7
    return gps_week, gps_dow


def decompress_file(filepath: str, remove_compressed: bool = True) -> bool:
    """
    Decompress .Z or .gz file
    
    Parameters
    ----------
    filepath : str
        Path to compressed file
    remove_compressed : bool
        Remove compressed file after decompression
        
    Returns
    -------
    bool
        True if successful
    """
    extension = os.path.splitext(filepath)[1]
    decompressed_path = os.path.splitext(filepath)[0]
    
    try:
        if extension == '.gz':
            with gzip.open(filepath, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif extension == '.Z':
            # Try using system uncompress/gunzip
            try:
                subprocess.run(['gunzip', '-f', filepath], check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Try uncompress
                try:
                    subprocess.run(['uncompress', '-f', filepath], check=True)
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Try using Python unlzw if available
                    try:
                        import unlzw3
                        with open(filepath, 'rb') as f_in:
                            with open(decompressed_path, 'wb') as f_out:
                                f_out.write(unlzw3.unlzw(f_in.read()))
                    except ImportError:
                        print("Cannot decompress .Z file - no decompressor available")
                        return False
        
        if remove_compressed and os.path.exists(filepath):
            os.remove(filepath)
        
        return os.path.exists(decompressed_path)
        
    except Exception as e:
        print(f"Decompression error: {e}")
        return False


def download_sp3_ftp(date: datetime, product: str = "igs", 
                     cache_dir: str = "./sp3_cache",
                     overwrite: bool = False) -> Optional[str]:
    """
    Download SP3 file via FTP
    
    Parameters
    ----------
    date : datetime
        Date for which to download SP3 file
    product : str
        Product type: 'igs', 'igr', 'igu', 'cod', 'gfz', 'wum'
    cache_dir : str
        Directory to cache downloaded files
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
    
    # Build filename and path based on product
    if product in ["igs", "igr", "igu"]:
        # Standard IGS products - try both old and new naming conventions
        if product == "igs":
            # Try new long name format first
            longname = f"IGS0OPSFIN_{year:04d}{doy:03d}0000_01D_15M_ORB.SP3"
            shortname = f"igs{gps_week:04d}{gps_dow:01d}.sp3"
            servers = [
                ("igs-ftp.bkg.bund.de", f"/IGS/products/{gps_week:04d}/{longname}.gz"),
                ("igs-ftp.bkg.bund.de", f"/IGS/products/{gps_week:04d}/{shortname}.Z"),
                ("gdc.cddis.eosdis.nasa.gov", f"/gnss/products/{gps_week:04d}/{shortname}.Z"),
            ]
            filename = shortname
        elif product == "igr":
            # IGS rapid
            longname = f"IGS0OPSRAP_{year:04d}{doy:03d}0000_01D_15M_ORB.SP3"
            shortname = f"igr{gps_week:04d}{gps_dow:01d}.sp3"
            servers = [
                ("igs-ftp.bkg.bund.de", f"/IGS/products/{gps_week:04d}/{longname}.gz"),
                ("igs-ftp.bkg.bund.de", f"/IGS/products/{gps_week:04d}/{shortname}.Z"),
                ("gdc.cddis.eosdis.nasa.gov", f"/gnss/products/{gps_week:04d}/{shortname}.Z"),
            ]
            filename = shortname
        else:  # igu
            hour = (date.hour // 6) * 6
            longname = f"IGS0OPSULT_{year:04d}{doy:03d}{hour:02d}00_01D_15M_ORB.SP3"
            shortname = f"igu{gps_week:04d}{gps_dow:01d}_{hour:02d}.sp3"
            servers = [
                ("igs-ftp.bkg.bund.de", f"/IGS/products/{gps_week:04d}/{longname}.gz"),
                ("igs-ftp.bkg.bund.de", f"/IGS/products/{gps_week:04d}/{shortname}.Z"),
                ("gdc.cddis.eosdis.nasa.gov", f"/gnss/products/{gps_week:04d}/{shortname}.Z"),
            ]
            filename = shortname
        
    elif product in ["cod", "gfz", "wum"]:
        # MGEX products with long names
        if product == "cod":
            longname = f"COD0MGXFIN_{year:04d}{doy:03d}0000_01D_05M_ORB.SP3"
            shortname = f"cod{gps_week:04d}{gps_dow:01d}.sp3"
        elif product == "gfz":
            longname = f"GFZ0MGXRAP_{year:04d}{doy:03d}0000_01D_05M_ORB.SP3"
            shortname = f"gfz{gps_week:04d}{gps_dow:01d}.sp3"
        elif product == "wum":
            longname = f"WUM0MGXFIN_{year:04d}{doy:03d}0000_01D_05M_ORB.SP3"
            shortname = f"wum{gps_week:04d}{gps_dow:01d}.sp3"
        
        # Try both long and short names
        servers = [
            ("igs-ftp.bkg.bund.de", f"/IGS/products/mgex/{gps_week:04d}/{longname}.gz"),
            ("gdc.cddis.eosdis.nasa.gov", f"/gnss/products/mgex/{gps_week:04d}/{longname}.gz"),
            ("igs-ftp.bkg.bund.de", f"/IGS/products/{gps_week:04d}/{shortname}.Z"),
            ("gdc.cddis.eosdis.nasa.gov", f"/gnss/products/{gps_week:04d}/{shortname}.Z"),
        ]
        filename = shortname  # Use short name for local file
    else:
        print(f"Unknown product: {product}")
        return None
    
    # Check if file already exists
    local_path = cache_path / filename
    if local_path.exists() and not overwrite:
        print(f"SP3 file already exists: {local_path}")
        return str(local_path)
    
    # Try each server
    for server_url, ftp_path in servers:
        print(f"Trying FTP server: {server_url}{ftp_path}")
        
        try:
            # Determine if secure FTP is needed
            secure = (server_url == "gdc.cddis.eosdis.nasa.gov")
            
            # Connect to FTP
            if secure:
                ftp = FTP_TLS(server_url, timeout=30)
                ftp.login()  # Anonymous login
                ftp.prot_p()  # Enable protection
            else:
                ftp = FTP(server_url, timeout=30)
                ftp.login()  # Anonymous login
            
            # Download file
            compressed_ext = ".gz" if ftp_path.endswith(".gz") else ".Z"
            compressed_path = local_path.with_suffix(local_path.suffix + compressed_ext)
            
            with open(compressed_path, 'wb') as f:
                ftp.retrbinary(f'RETR {ftp_path}', f.write)
            
            ftp.quit()
            print(f"Downloaded: {compressed_path}")
            
            # Decompress
            if decompress_file(str(compressed_path), remove_compressed=True):
                if local_path.exists():
                    print(f"Successfully downloaded and decompressed: {local_path}")
                    return str(local_path)
            
        except Exception as e:
            print(f"Failed to download from {server_url}: {e}")
            continue
    
    print(f"Failed to download SP3 file for {date} from all servers")
    return None


def download_clk_ftp(date: datetime, product: str = "igs",
                     cache_dir: str = "./sp3_cache",
                     overwrite: bool = False) -> Optional[str]:
    """
    Download CLK file via FTP
    
    Similar to SP3 but for clock files
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    gps_week, gps_dow = gps_week_day(date)
    year = date.year
    doy = date.timetuple().tm_yday
    
    # Build filename based on product
    if product in ["igs", "igr"]:
        filename = f"{product}{gps_week:04d}{gps_dow:01d}.clk"
        servers = [
            ("igs-ftp.bkg.bund.de", f"/IGS/products/{gps_week:04d}/{filename}.Z"),
            ("gdc.cddis.eosdis.nasa.gov", f"/gnss/products/{gps_week:04d}/{filename}.Z"),
        ]
    elif product in ["cod", "gfz", "wum"]:
        if product == "cod":
            longname = f"COD0MGXFIN_{year:04d}{doy:03d}0000_01D_30S_CLK.CLK"
            shortname = f"cod{gps_week:04d}{gps_dow:01d}.clk"
        elif product == "gfz":
            longname = f"GFZ0MGXRAP_{year:04d}{doy:03d}0000_01D_30S_CLK.CLK"
            shortname = f"gfz{gps_week:04d}{gps_dow:01d}.clk"
        elif product == "wum":
            longname = f"WUM0MGXFIN_{year:04d}{doy:03d}0000_01D_30S_CLK.CLK"
            shortname = f"wum{gps_week:04d}{gps_dow:01d}.clk"
        
        servers = [
            ("igs-ftp.bkg.bund.de", f"/IGS/products/mgex/{gps_week:04d}/{longname}.gz"),
            ("gdc.cddis.eosdis.nasa.gov", f"/gnss/products/mgex/{gps_week:04d}/{longname}.gz"),
            ("igs-ftp.bkg.bund.de", f"/IGS/products/{gps_week:04d}/{shortname}.Z"),
            ("gdc.cddis.eosdis.nasa.gov", f"/gnss/products/{gps_week:04d}/{shortname}.Z"),
        ]
        filename = shortname
    else:
        print(f"Unknown product: {product}")
        return None
    
    # Check if file already exists
    local_path = cache_path / filename
    if local_path.exists() and not overwrite:
        print(f"CLK file already exists: {local_path}")
        return str(local_path)
    
    # Try each server (similar to SP3)
    for server_url, ftp_path in servers:
        print(f"Trying FTP server: {server_url}{ftp_path}")
        
        try:
            secure = (server_url == "gdc.cddis.eosdis.nasa.gov")
            
            if secure:
                ftp = FTP_TLS(server_url, timeout=30)
                ftp.login()
                ftp.prot_p()
            else:
                ftp = FTP(server_url, timeout=30)
                ftp.login()
            
            compressed_ext = ".gz" if ftp_path.endswith(".gz") else ".Z"
            compressed_path = local_path.with_suffix(local_path.suffix + compressed_ext)
            
            with open(compressed_path, 'wb') as f:
                ftp.retrbinary(f'RETR {ftp_path}', f.write)
            
            ftp.quit()
            print(f"Downloaded: {compressed_path}")
            
            if decompress_file(str(compressed_path), remove_compressed=True):
                if local_path.exists():
                    print(f"Successfully downloaded and decompressed: {local_path}")
                    return str(local_path)
            
        except Exception as e:
            print(f"Failed to download from {server_url}: {e}")
            continue
    
    print(f"Failed to download CLK file for {date} from all servers")
    return None


def get_best_sp3_product(date: datetime, cache_dir: str = "./sp3_cache") -> Optional[str]:
    """
    Download the best available SP3 product for a given date
    
    Following gnss_lib_py strategy:
    - Within 3 days: Try CODE rapid (COD0OPSRAP)
    - Within 2 weeks: Try GFZ rapid (GFZ0MGXRAP)
    - Older: Try IGS/CODE/WUM final products
    """
    now = datetime.now()
    age_days = (now - date).days
    
    print(f"Date: {date}, Age: {age_days} days")
    
    # Determine product priority based on age
    if age_days <= 3:
        # Very recent - try ultra-rapid and rapid products
        products = ['igu', 'igr', 'gfz', 'cod']
    elif age_days <= 14:
        # Recent - try rapid then final
        products = ['igr', 'gfz', 'cod', 'igs']
    elif age_days <= 35:
        # About a month old - rapid should be available
        products = ['igr', 'igs', 'cod', 'gfz', 'wum']
    else:
        # Older - try final products
        products = ['igs', 'cod', 'wum', 'gfz', 'igr']
    
    # Try each product
    for product in products:
        print(f"\nTrying {product.upper()} product...")
        sp3_file = download_sp3_ftp(date, product, cache_dir)
        if sp3_file:
            return sp3_file
    
    return None


def main():
    """Test SP3 download for a specific date"""
    
    # Test with 2024-01-15 (should definitely have final products)
    test_date = datetime(2024, 1, 15, 12, 0, 0)
    
    print("="*80)
    print(f"Testing SP3 download for: {test_date}")
    print("="*80)
    
    # Try to get best available product
    sp3_file = get_best_sp3_product(test_date, cache_dir="./sp3_cache")
    
    if sp3_file:
        print(f"\n✓ Successfully obtained SP3 file: {sp3_file}")
        
        # Also try to get clock file
        print("\nTrying to download corresponding CLK file...")
        clk_file = download_clk_ftp(test_date, product="igs", cache_dir="./sp3_cache")
        if clk_file:
            print(f"✓ Successfully obtained CLK file: {clk_file}")
    else:
        print("\n✗ Failed to download SP3 file")
    
    print("\n" + "="*80)
    print("Test complete!")


if __name__ == "__main__":
    main()