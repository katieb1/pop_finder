# Script for reading genetic data and population data

# Load packages
import numpy as np
import allel
import pandas as pd
import zarr
import h5py
import sys
import argparse

def read_data(infile, sample_data, save_allele_counts, kfcv=False):
    """
    Reads a .zarr, .vcf, or h5py file containing genetic data and 
    creates subsettable data for a classifier neural network.
    
    Parameters
    ----------
    infile : string
        Path to the .zarr, .vcf, or h5py file.
    sample_data : string
        Path to .txt file containing sample information
        (columns are x, y, sampleID, and pop).
    save_allele_counts : boolean
        Saves derived allele count information (Default=False).
    kfcv : boolean
        If being used to test accuracy with k-fold cross-
        validation (i.e. no NAs in the sample data), set to
        True (Default=False).
        
    Returns
    -------
    samp_list : dataframe
        Contains information on corresponding sampleID and 
        population classifications.
    dc : np.array
        Array of derived allele counts.
    unknowns : dataframe
        If kfcv is set to False, returns a dataframe with 
        information about sampleID and indices for samples
        of unknown origin.
    """
    
    # Load genotypes
    print("loading genotypes")
    if infile.endswith('.zarr'):
        
        callset = zarr.open_group(infile, mode='r')
        gt = callset['calldata/GT']
        gen = allel.GenotypeArray(gt[:])
        samples = callset['samples'][:]
        
    elif infile.endswith('.vcf') or infile.endswith('.vcf.gz'):
        
        vcf=allel.read_vcf(infile,log=sys.stderr)
        gen=allel.GenotypeArray(vcf['calldata/GT'])
        pos=vcf['variants/POS']
        samples=vcf['samples']
        
    elif infile.endswith('.locator.hdf5'):
        
        h5=h5py.File(infile,'r')
        dc=np.array(h5['derived_counts'])
        samples=np.array(h5['samples'])
        h5.close()
    
    #count derived alleles for biallelic sites
    if not infile.endswith('.locator.hdf5'):
        
        print("counting alleles")
        ac=gen.to_allele_counts()
        biallel=gen.count_alleles().is_biallelic()
        dc=np.array(ac[biallel,:,1],dtype="int_")
        dc=np.transpose(dc)
        
        if save_allele_counts and not infile.endswith('.locator.hdf5'):
            
            print("saving derived counts for reanalysis")
            outfile=h5py.File(infile+".locator.hdf5", "w")
            outfile.create_dataset("derived_counts", data=dc)
            outfile.create_dataset("samples", data=samples,dtype=h5py.string_dtype()) #note this requires h5py v 2.10.0
            outfile.close()
            #sys.exit()
        
    # Load data and organize for output
    print("loading sample data")
    locs=pd.read_csv(sample_data,sep="\t")
    locs['id']=locs['sampleID']
    locs.set_index('id',inplace=True)
    
    #sort loc table so samples are in same order as genotype samples
    locs=locs.reindex(np.array(samples)) 
    
    #check that all sample names are present
    if not all([locs['sampleID'][x]==samples[x] for x in range(len(samples))]):
        
        print("sample ordering failed! Check that sample IDs match the VCF.")
        sys.exit()
        
    if kfcv==True:
        
        locs=np.array(locs["pop"])
        samp_list = pd.DataFrame({'samples': samples,'pops': locs})
        
        # Return the sample list to be funneled into kfcv
        return samp_list, dc
    
    else:
        
        locs['order'] = np.arange(len(locs))
        
        # Find unknown locations as NAs in the dataset
        unknowns = locs.iloc[np.where(pd.isnull(locs['pop']))]
        
        # Extract known location information for training
        samples = samples[np.where(pd.notnull(locs['pop']))]
        locs = locs.iloc[np.where(pd.notnull(locs['pop']))]
        order = np.array(locs['order'])
        locs = np.array(locs["pop"])
        samp_list = pd.DataFrame({'samples': samples, 'pops': locs, 'order': order})
        
        return samp_list, dc, unknowns
        
        