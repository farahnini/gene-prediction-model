import os
import time
import logging
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class GenomeAnnotationDownloader:
    def __init__(self, output_dir: str = "genome_annotations", max_workers: int = 3):
        self.output_dir = output_dir
        self.max_workers = max_workers
        self._setup_logging()
        self._setup_directories()
        self.refseq_delay = 2
        self.last_refseq_request = 0
        self.ftp_dirs = {
            "Homo sapiens": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/",
            "Mus musculus": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/GCF_000001635.27_GRCm39/",
            "Rattus norvegicus": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/036/323/735/GCF_036323735.1_GRCr8/",
            "Pan troglodytes": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/028/858/775/GCF_028858775.2_NHGRI_mPanTro3-v2.0_pri/",
            "Gallus gallus": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/016/699/485/GCF_016699485.2_bGalGal1.mat.broiler.GRCg7b/",
            "Taeniopygia guttata": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/048/771/995/GCF_048771995.1_bTaeGut7.mat/",
            "Meleagris gallopavo": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/605/GCF_000146605.3_Turkey_5.1/",
            "Danio rerio": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/049/306/965/GCF_049306965.1_GRCz12tu/",
            "Oryzias latipes": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/002/234/675/GCF_002234675.1_ASM223467v1/",
            "Gasterosteus aculeatus": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/016/920/845/GCF_016920845.1_GAculeatus_UGA_version5/",
            "Drosophila melanogaster": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/215/GCF_000001215.4_Release_6_plus_ISO1_MT/",
            "Apis mellifera": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/003/254/395/GCF_003254395.2_Amel_HAv3.1/",
            "Bombyx mori": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/030/269/925/GCF_030269925.1_ASM3026992v2/",
            "Arabidopsis thaliana": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/735/GCF_000001735.4_TAIR10.1/",
            "Oryza sativa": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/034/140/825/GCF_034140825.1_ASM3414082v1/",
            "Zea mays": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/902/167/145/GCF_902167145.1_Zm-B73-REFERENCE-NAM-5.0/",
            "Glycine max": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/004/515/GCF_000004515.6_Glycine_max_v4.0/",
            "Saccharomyces cerevisiae": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/GCF_000146045.2_R64/",
            "Schizosaccharomyces pombe": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/002/945/GCF_000002945.2_ASM294v3/",
            "Aspergillus nidulans": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/011/425/GCF_000011425.1_ASM1142v1/",
            # Bacteria
            "Escherichia coli": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/", 
            "Bacillus subtilis": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/009/045/GCF_000009045.1_ASM904v1/", 
            "Pseudomonas aeruginosa": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/765/GCF_000006765.1_ASM676v1/",  
            "Mycobacterium tuberculosis": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/195/955/GCF_000195955.2_ASM19595v2/", 
            # Archaea
            "Methanocaldococcus jannaschii": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/091/665/GCF_000091665.1_ASM9166v1/", 
            "Sulfolobus solfataricus": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/007/765/GCF_000007765.1_ASM776v1/",  
            "Halobacterium salinarum": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/805/GCF_000006805.1_ASM680v1/",  
            # Nematodes
            "Caenorhabditis elegans": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/002/985/GCF_000002985.6_WBcel235/",  
            "Caenorhabditis briggsae": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/004/555/GCF_000004555.2_CB4/" ,  
             # Amphibians
            "Xenopus tropicalis": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/004/195/GCF_000004195.4_UCB_Xtro_10.0/",
            "Xenopus laevis": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/001/663/975/GCF_001663975.1_Xenopus_laevis_v2/",
            # Reptiles
            "Anolis carolinensis": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/090/745/GCF_000090745.1_AnoCar2.0/",  
            "Python bivittatus": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/186/305/GCF_000186305.1_Python_molurus_bivittatus-5.0.2/",
        }
        self.training_species = list(self.ftp_dirs.keys())
        self.download_status = {
            'refseq': {},
            'last_updated': None
        }

    def _setup_logging(self):
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"download_{time.strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _setup_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "refseq"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "fasta"), exist_ok=True)

    def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_refseq_request
        if time_since_last < self.refseq_delay:
            time.sleep(self.refseq_delay - time_since_last)
        self.last_refseq_request = time.time()

    def _download_annotation(self, species: str) -> bool:
        try:
            self._rate_limit()
            if species in self.ftp_dirs:
                ftp_dir = self.ftp_dirs[species]
                basename = ftp_dir.rstrip('/').split('/')[-1]
                url = f"{ftp_dir}{basename}_genomic.gff.gz"
                output_file = os.path.join(self.output_dir, "refseq", f"{species.replace(' ', '_')}.gff.gz")
                if os.path.exists(output_file):
                    logging.info(f"Annotation file for {species} already exists, skipping download.")
                    self.download_status['refseq'][species] = True
                    return True
                r = requests.get(url, stream=True)
                if r.status_code == 200:
                    with open(output_file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    logging.info(f"Successfully downloaded annotation for {species} from {url}")
                    self.download_status['refseq'][species] = True
                    return True
                else:
                    logging.error(f"Failed to download annotation for {species} from {url} (HTTP {r.status_code})")
                    self.download_status['refseq'][species] = False
                    return False
            else:
                logging.error(f"No FTP directory mapping for {species}. Please add it to ftp_dirs.")
                self.download_status['refseq'][species] = False
                return False
        except Exception as e:
            logging.error(f"Error downloading annotation for {species}: {str(e)}")
            self.download_status['refseq'][species] = False
            return False

    def _download_fasta(self, species: str) -> bool:
        try:
            self._rate_limit()
            if species in self.ftp_dirs:
                ftp_dir = self.ftp_dirs[species]
                basename = ftp_dir.rstrip('/').split('/')[-1]
                url = f"{ftp_dir}{basename}_genomic.fna.gz"
                output_file = os.path.join(self.output_dir, "fasta", f"{species.replace(' ', '_')}.fna.gz")
                if os.path.exists(output_file):
                    logging.info(f"FASTA file for {species} already exists, skipping download.")
                    return True
                r = requests.get(url, stream=True)
                if r.status_code == 200:
                    with open(output_file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    logging.info(f"Successfully downloaded FASTA for {species} from {url}")
                    return True
                else:
                    logging.error(f"Failed to download FASTA for {species} from {url} (HTTP {r.status_code})")
                    return False
            else:
                logging.error(f"No FTP directory mapping for {species}. Please add it to ftp_dirs.")
                return False
        except Exception as e:
            logging.error(f"Error downloading FASTA for {species}: {str(e)}")
            return False

    def download_all_annotations(self):
        total_species = len(self.training_species)
        logging.info("Starting annotation and FASTA downloads (direct FTP only)...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for species in self.training_species:
                logging.info(f"Processing {species} ...")
                futures.append(executor.submit(self._download_annotation, species))
                futures.append(executor.submit(self._download_fasta, species))
            for future in tqdm(as_completed(futures), total=2*total_species, desc="Downloading Annotations and FASTA"):
                future.result()
        self._save_download_status()
        self._generate_summary()

    def _save_download_status(self):
        status_file = os.path.join(self.output_dir, "download_status.json")
        self.download_status['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(status_file, 'w') as f:
            import json
            json.dump(self.download_status, f, indent=4)
        logging.info(f"Download status saved to {status_file}")

    def _generate_summary(self):
        logging.info("\nDownload Summary:")
        logging.info("-" * 50)
        success_count = sum(1 for status in self.download_status['refseq'].values() if status)
        total_count = len(self.download_status['refseq'])
        logging.info(f"REFSEQ: {success_count}/{total_count} successful downloads")
        logging.info("-" * 50)

def main():
    downloader = GenomeAnnotationDownloader()
    downloader.download_all_annotations()

if __name__ == "__main__":
    main()
