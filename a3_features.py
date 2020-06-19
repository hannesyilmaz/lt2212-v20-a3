import os
import sys
import argparse
import numpy as np
import pandas as pd
# Whatever other imports you need

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    complete_files, author_contents_complete, authors = read_documents(args.inputdir)

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    table_df = build_table(args.dims, args.testsize, author_contents_complete, authors)
    
    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    table_df.to_csv(args.outputfile)

    print("Done!")
    
