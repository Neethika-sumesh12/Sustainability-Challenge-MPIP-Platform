"""
NIST NIR Spectroscopy Data Exploration
Load, summarize, and visualize polyolefin NIR spectra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Set paths
data_dir = Path("NIR_property_prediction/Data")
nir_dir = data_dir / "NIR"
property_file = data_dir / "PropertyMeasurements.csv"
sample_info_file = data_dir / "SampleInformation.csv"

print("=" * 80)
print("NIST NIR Polyolefin Dataset Exploration")
print("=" * 80)

# Step 1: Load all NIR spectra and average replicates
print("\n[1] Loading NIR spectra...")

# Get all CSV files in NIR directory
nir_files = sorted(glob.glob(str(nir_dir / "*.csv")))
print(f"Found {len(nir_files)} NIR spectrum files")

# Dictionary to store spectra by sample
spectra_by_sample = {}

for file_path in nir_files:
    # Extract sample name (e.g., "H0007HDPE" from "H0007HDPE_1.csv")
    filename = Path(file_path).stem
    sample_name = filename.rsplit('_', 1)[0]  # Remove replicate number
    
    # Load spectrum
    spectrum = pd.read_csv(file_path)
    
    # Store in dictionary
    if sample_name not in spectra_by_sample:
        spectra_by_sample[sample_name] = []
    spectra_by_sample[sample_name].append(spectrum)

print(f"Found {len(spectra_by_sample)} unique samples")

# Average replicates for each sample
mean_spectra = {}
for sample_name, replicate_list in spectra_by_sample.items():
    # Stack all replicates and compute mean
    stacked = pd.concat(replicate_list, axis=1)
    # Get wavenumber from first column
    wavenumber = stacked.iloc[:, 0]
    # Average all intensity columns
    intensity_cols = [col for i, col in enumerate(stacked.columns) if i % 2 == 1]
    mean_intensity = stacked[intensity_cols].mean(axis=1)
    
    mean_spectra[sample_name] = pd.DataFrame({
        'wavenumber': wavenumber,
        'intensity': mean_intensity
    })

print(f"Averaged {len(mean_spectra)} samples (6 replicates each → 1 mean spectrum)")

# Step 2: Load property measurements
print("\n[2] Loading property measurements...")
properties = pd.read_csv(property_file)
print(f"Loaded properties for {len(properties)} samples")
print(f"Properties: {list(properties.columns)}")

# Step 3: Extract polymer class from sample label
print("\n[3] Extracting polymer classes from sample names...")

def extract_polymer_class(label):
    """Extract polymer type from sample label"""
    label_upper = label.upper()
    if 'HDPE' in label_upper:
        return 'HDPE'
    elif 'LDPE' in label_upper and 'L_' not in label_upper:
        return 'LDPE'
    elif 'LLDPE' in label_upper:
        return 'LLDPE'
    elif 'ULDPE' in label_upper:
        return 'ULDPE'
    elif 'MDPE' in label_upper:
        return 'MDPE'
    elif 'PP' in label_upper and 'RPP' not in label_upper:
        return 'PP'
    elif 'RPP' in label_upper:
        return 'RPP (Recycled PP)'
    elif 'H_' in label_upper or 'L_' in label_upper:
        return 'Blend (HDPE+PP or LDPE+HDPE)'
    else:
        return 'Unknown'

properties['polymer_class'] = properties['label'].apply(extract_polymer_class)

# Step 4: Print summary table
print("\n[4] Summary Table:")
print("=" * 100)
print(f"{'Sample Label':<20} {'Polymer Class':<25} {'Density (g/cm³)':<18} {'Crystallinity':<15} {'SCB (CH3/1000C)':<15}")
print("=" * 100)

for idx, row in properties.iterrows():
    print(f"{row['label']:<20} {row['polymer_class']:<25} {row['density [g/cm3]']:<18.4f} {row['crystallinity']:<15.4f} {row['SCB [CH3/1000C]']:<15.2f}")

print("=" * 100)

# Print statistics by polymer class
print("\n[5] Statistics by Polymer Class:")
print("=" * 80)
class_stats = properties.groupby('polymer_class').agg({
    'density [g/cm3]': ['count', 'mean', 'std'],
    'crystallinity': ['mean', 'std'],
    'SCB [CH3/1000C]': ['mean', 'std']
})
print(class_stats.round(4))

# Step 5: Plot comparison of HDPE vs LDPE spectra
print("\n[6] Plotting HDPE vs LDPE spectra comparison...")

# Find one HDPE and one LDPE sample
hdpe_samples = [s for s in mean_spectra.keys() if 'HDPE' in s and 'LDPE' not in s]
ldpe_samples = [s for s in mean_spectra.keys() if 'LDPE' in s and 'LLDPE' not in s and 'ULDPE' not in s]

if hdpe_samples and ldpe_samples:
    hdpe_sample = hdpe_samples[0]
    ldpe_sample = ldpe_samples[0]
    
    # Get spectra
    hdpe_spectrum = mean_spectra[hdpe_sample]
    ldpe_spectrum = mean_spectra[ldpe_sample]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(hdpe_spectrum['wavenumber'], hdpe_spectrum['intensity'], 
             label=f'HDPE ({hdpe_sample})', linewidth=1.5, alpha=0.8)
    plt.plot(ldpe_spectrum['wavenumber'], ldpe_spectrum['intensity'], 
             label=f'LDPE ({ldpe_sample})', linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    plt.ylabel('Reflectance Intensity', fontsize=12)
    plt.title('NIR Spectra Comparison: HDPE vs LDPE\n(Showing spectral similarity - the core challenge for NIR sorting)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('nir_hdpe_vs_ldpe_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved as 'nir_hdpe_vs_ldpe_comparison.png'")
    print(f"  Comparing: {hdpe_sample} vs {ldpe_sample}")
    
    # Show plot
    plt.show()
else:
    print("⚠ Could not find both HDPE and LDPE samples for comparison")

print("\n" + "=" * 80)
print("Data exploration complete!")
print("=" * 80)
print(f"\nSummary:")
print(f"  • Total samples: {len(properties)}")
print(f"  • Total spectra files: {len(nir_files)}")
print(f"  • Averaged to: {len(mean_spectra)} mean spectra")
print(f"  • Wavenumber range: {hdpe_spectrum['wavenumber'].min():.0f} - {hdpe_spectrum['wavenumber'].max():.0f} cm⁻¹")
print(f"  • Data points per spectrum: {len(hdpe_spectrum)}")
print("\nNext steps: Build property prediction models")

# Made with Bob
