import os
import requests
import pandas as pd
from Bio import Entrez
from tqdm import tqdm
import gzip
import shutil
from datetime import datetime
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class GenomeAnnotationDownloader:
    def __init__(self, email, output_dir="genome_annotations", max_workers=4):
        """
        Initialize the downloader with your email (required by NCBI) and output directory.
        
        Args:
            email (str): Your email address (required by NCBI)
            output_dir (str): Directory to save downloaded annotations
            max_workers (int): Maximum number of parallel downloads
        """
        self.email = email
        self.output_dir = output_dir
        self.max_workers = max_workers
        Entrez.email = email
        
        # Setup logging
        self._setup_logging()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load species information
        self.training_species = self._load_species_info()
        
        # Track download progress
        self.download_status = {
            'refseq': {},
            'ensembl': {},
            'failed': []
        }
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_species_info(self):
        """Load species information from a structured dictionary."""
        return {
            # Mammals
            "Homo sapiens": {"common_name": "Human", "lineage": "Mammals"},
            "Mus musculus": {"common_name": "Mouse", "lineage": "Mammals"},
            "Rattus norvegicus": {"common_name": "Rat", "lineage": "Mammals"},
            "Pan troglodytes": {"common_name": "Chimpanzee", "lineage": "Mammals"},
            
            # Birds
            "Gallus gallus": {"common_name": "Chicken", "lineage": "Birds"},
            "Taeniopygia guttata": {"common_name": "Zebra finch", "lineage": "Birds"},
            "Meleagris gallopavo": {"common_name": "Turkey", "lineage": "Birds"},
            
            # Fish
            "Danio rerio": {"common_name": "Zebrafish", "lineage": "Fish"},
            "Oryzias latipes": {"common_name": "Japanese medaka", "lineage": "Fish"},
            "Gasterosteus aculeatus": {"common_name": "Stickleback", "lineage": "Fish"},
            
            # Insects
            "Drosophila melanogaster": {"common_name": "Fruit fly", "lineage": "Insects"},
            "Apis mellifera": {"common_name": "Honey bee", "lineage": "Insects"},
            "Bombyx mori": {"common_name": "Silkworm", "lineage": "Insects"},
            
            # Plants
            "Arabidopsis thaliana": {"common_name": "Thale cress", "lineage": "Plants"},
            "Oryza sativa": {"common_name": "Rice", "lineage": "Plants"},
            "Zea mays": {"common_name": "Maize", "lineage": "Plants"},
            "Glycine max": {"common_name": "Soybean", "lineage": "Plants"},
            
            # Fungi
            "Saccharomyces cerevisiae": {"common_name": "Baker's yeast", "lineage": "Fungi"},
            "Schizosaccharomyces pombe": {"common_name": "Fission yeast", "lineage": "Fungi"},
            "Aspergillus nidulans": {"common_name": "Aspergillus", "lineage": "Fungi"},
            
            # Bacteria
            "Escherichia coli": {"common_name": "E. coli", "lineage": "Bacteria"},
            "Bacillus subtilis": {"common_name": "B. subtilis", "lineage": "Bacteria"},
            "Pseudomonas aeruginosa": {"common_name": "P. aeruginosa", "lineage": "Bacteria"},
            "Mycobacterium tuberculosis": {"common_name": "M. tuberculosis", "lineage": "Bacteria"},
            
            # Archaea
            "Methanocaldococcus jannaschii": {"common_name": "Methanococcus", "lineage": "Archaea"},
            "Sulfolobus solfataricus": {"common_name": "Sulfolobus", "lineage": "Archaea"},
            "Halobacterium salinarum": {"common_name": "Halobacterium", "lineage": "Archaea"},
            
            # Nematodes
            "Caenorhabditis elegans": {"common_name": "C. elegans", "lineage": "Nematodes"},
            "Caenorhabditis briggsae": {"common_name": "C. briggsae", "lineage": "Nematodes"},
            
            # Amphibians
            "Xenopus tropicalis": {"common_name": "Western clawed frog", "lineage": "Amphibians"},
            "Xenopus laevis": {"common_name": "African clawed frog", "lineage": "Amphibians"},
            
            # Reptiles
            "Anolis carolinensis": {"common_name": "Green anole", "lineage": "Reptiles"},
            "Python bivittatus": {"common_name": "Burmese python", "lineage": "Reptiles"}
        }
    
    def _create_species_directory(self, species):
        """Create directory structure for a species."""
        species_dir = os.path.join(self.output_dir, species.replace(" ", "_"))
        os.makedirs(species_dir, exist_ok=True)
        return species_dir
    
    def _download_refseq_annotation(self, species, info):
        """Download RefSeq annotation for a single species."""
        try:
            self.logger.info(f"Processing {species} ({info['common_name']}) for RefSeq...")
            
            # Search for the genome
            search_term = f'"{species}"[Organism] AND RefSeq[Source] AND latest[RefSeq Status]'
            handle = Entrez.esearch(db="genome", term=search_term)
            record = Entrez.read(handle)
            handle.close()
            
            if not record["IdList"]:
                self.logger.warning(f"No RefSeq genome found for {species}")
                return False
            
            genome_id = record["IdList"][0]
            
            # Create species directory
            species_dir = self._create_species_directory(species)
            
            # Download GFF annotation file
            handle = Entrez.efetch(db="genome", id=genome_id, rettype="gff", retmode="text")
            gff_content = handle.read()
            handle.close()
            
            # Save GFF file
            gff_file = os.path.join(species_dir, f"{species.replace(' ', '_')}_refseq.gff")
            with open(gff_file, "w") as f:
                f.write(gff_content)
            
            self.logger.info(f"Successfully downloaded RefSeq annotations for {species}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading RefSeq for {species}: {str(e)}")
            return False
    
    def _download_ensembl_annotation(self, species, info):
        """Download Ensembl annotation for a single species."""
        try:
            self.logger.info(f"Processing {species} ({info['common_name']}) for Ensembl...")
            
            # Create species directory
            species_dir = self._create_species_directory(species)
            
            # Format species name for Ensembl
            ensembl_species = species.lower().replace(" ", "_")
            
            # Get latest release
            release = self._get_latest_ensembl_release()
            
            # Download GFF annotation file
            gff_url = f"https://ftp.ensembl.org/pub/release-{release}/gff3/{ensembl_species}/"
            gff_file = f"{ensembl_species.capitalize()}.{release}.gff3.gz"
            
            response = requests.get(gff_url + gff_file, stream=True)
            if response.status_code == 200:
                # Save compressed file
                gz_file = os.path.join(species_dir, gff_file)
                with open(gz_file, "wb") as f:
                    f.write(response.content)
                
                # Decompress file
                with gzip.open(gz_file, "rb") as f_in:
                    with open(gz_file[:-3], "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove compressed file
                os.remove(gz_file)
                self.logger.info(f"Successfully downloaded Ensembl annotations for {species}")
                return True
            else:
                self.logger.warning(f"No Ensembl annotations found for {species}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading Ensembl for {species}: {str(e)}")
            return False
    
    def download_refseq_annotations(self):
        """Download genome annotations from NCBI RefSeq for all species."""
        self.logger.info("Starting RefSeq downloads...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_species = {
                executor.submit(self._download_refseq_annotation, species, info): species
                for species, info in self.training_species.items()
            }
            
            for future in tqdm(as_completed(future_to_species), total=len(future_to_species), desc="Downloading RefSeq"):
                species = future_to_species[future]
                try:
                    success = future.result()
                    self.download_status['refseq'][species] = success
                except Exception as e:
                    self.logger.error(f"Error processing {species}: {str(e)}")
                    self.download_status['failed'].append(species)
    
    def download_ensembl_annotations(self):
        """Download genome annotations from Ensembl for all species."""
        self.logger.info("Starting Ensembl downloads...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_species = {
                executor.submit(self._download_ensembl_annotation, species, info): species
                for species, info in self.training_species.items()
            }
            
            for future in tqdm(as_completed(future_to_species), total=len(future_to_species), desc="Downloading Ensembl"):
                species = future_to_species[future]
                try:
                    success = future.result()
                    self.download_status['ensembl'][species] = success
                except Exception as e:
                    self.logger.error(f"Error processing {species}: {str(e)}")
                    self.download_status['failed'].append(species)
    
    def _get_latest_ensembl_release(self):
        """Get the latest Ensembl release number."""
        response = requests.get("https://rest.ensembl.org/info/data/")
        data = response.json()
        return data["releases"][-1]
    
    def save_download_status(self):
        """Save download status to a JSON file."""
        status_file = os.path.join(self.output_dir, 'download_status.json')
        with open(status_file, 'w') as f:
            json.dump(self.download_status, f, indent=4)
        self.logger.info(f"Download status saved to {status_file}")
    
    def generate_summary(self):
        """Generate a summary of the download process."""
        total_species = len(self.training_species)
        refseq_success = sum(1 for success in self.download_status['refseq'].values() if success)
        ensembl_success = sum(1 for success in self.download_status['ensembl'].values() if success)
        failed = len(self.download_status['failed'])
        
        summary = f"""
Download Summary:
----------------
Total species: {total_species}
RefSeq downloads successful: {refseq_success}
Ensembl downloads successful: {ensembl_success}
Failed downloads: {failed}

Failed species:
{chr(10).join(self.download_status['failed']) if self.download_status['failed'] else 'None'}
"""
        self.logger.info(summary)
        
        # Save summary to file
        summary_file = os.path.join(self.output_dir, 'download_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(summary)
        self.logger.info(f"Summary saved to {summary_file}")

def main():
    # Example usage
    email = "your.email@example.com"  # Replace with your email
    downloader = GenomeAnnotationDownloader(email)
    
    try:
        # Download annotations from both sources
        downloader.download_refseq_annotations()
        downloader.download_ensembl_annotations()
        
        # Save status and generate summary
        downloader.save_download_status()
        downloader.generate_summary()
        
    except KeyboardInterrupt:
        print("\nDownload interrupted by user. Saving progress...")
        downloader.save_download_status()
        downloader.generate_summary()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        downloader.save_download_status()
        downloader.generate_summary()

if __name__ == "__main__":
    main() 