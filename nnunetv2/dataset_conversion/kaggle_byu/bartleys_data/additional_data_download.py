import concurrent.futures
import glob
import os
import shutil
import threading
from typing import List, Tuple, Set

import blosc2
import numpy as np
import zarr
from batchgenerators.utilities.file_and_folder_operations import save_json, maybe_mkdir_p, join, load_json
from cryoet_data_portal import Client, Dataset, Run


class CziiCollector():
    """
    This class was used to redownload Brendan's data and convert his annotations to the original image shape.
    You don't need to use this again. Please use RawDatasetDownloader() instead!
    """
    def __init__(
            self,
            tmp_dir: str = "/media/isensee/data/kaggle_byu_additional_data/tmp/",
            out_dir: str = "/media/isensee/data/kaggle_byu_additional_data/data/",
            labels_csv: str = '/home/isensee/git_repos/nnu-netv2/nnunetv2/dataset_conversion/kaggle_byu/labels.csv',
            dataset_author: str = 'Yi-Wei Chang',
            num_threads: int = 3
    ):
        super().__init__()
        self.tmp_dir = tmp_dir
        self.out_dir = out_dir
        self.dataset_author = dataset_author
        self.num_threads = num_threads

        self.labels = np.loadtxt(labels_csv, delimiter=',', skiprows=1, dtype=str) if labels_csv is not None else None

        # Tmp dir
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir)

        # Out dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def process_tomogram(self, x, tomo_id):
        assert self.labels is not None, "Please provide a labels_csv"
        cols = np.where(self.labels[:, 3] == tomo_id)[0]
        if len(cols) > 0:
            annotations_made_in_shape = np.array((128, 512, 512))
            coords = self.labels[cols, :3].astype(float)

            # we need to revert the coordinates of z
            z_coords = coords[:, 0].astype(float)
            # revert z slicing
            indices = list(np.linspace(0, x.shape[0] - 1, 128).astype(int))
            z_coords_orig = np.array([indices[int(z)] for z in z_coords]).reshape((len(coords), -1))

            # now convert xy of the annotations
            xy_coords = coords[:, 1:3]
            orig_anno = xy_coords * ((np.array(x.shape[1:]) - 1) / (annotations_made_in_shape[1:] - 1))
            orig_anno = np.hstack((z_coords_orig, orig_anno))
            orig_anno = [list(i) for i in orig_anno]
        else:
            orig_anno = None

        return x, orig_anno

    def run(self):
        # ========== Query Datasets ==========
        client = Client()

        # Datasets by Author
        ds_all = Dataset.find(client, [Dataset.authors.name == self.dataset_author])
        print("=" * 25)
        print("N_DATASETS:", len(ds_all))
        print("=" * 25)

        # ========= Process Each Dataset ==========
        for ds in ds_all:
            s = "TOTAL: {:<10}     TITLE: {}".format(len(ds.runs), ds.title)
            print(s)

            # Create dataset directory
            ds_dir = os.path.join(self.out_dir, str(ds.id))
            if not os.path.exists(ds_dir):
                os.mkdir(ds_dir)

            # Define a helper function to process a single run
            def process_run(run):
                outpath = os.path.join(ds_dir, str(run.name)) + ".b2nd"
                if os.path.exists(outpath[:-5] + '.json'):
                    print(f'skipping {run.name}')
                    return

                my_temp = join(self.tmp_dir, run.name)
                maybe_mkdir_p(my_temp)
                try:
                    # Download tomo
                    if len(run.tomograms) == 0:
                        print(f'run {run.name} has no tomograms')
                        return
                    tomo = run.tomograms[0]
                    tomo.download_omezarr(dest_path=my_temp)

                    # Load tomo
                    fpath = glob.glob(join(my_temp, "*"))[0]
                    arr = zarr.open(fpath, mode='r')
                    spacing = arr.attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
                    assert spacing[0] == tomo.voxel_spacing
                    arr = arr['0']

                    # Preprocess
                    arr, anno = self.process_tomogram(arr, run.name)

                    # Save
                    clevel: int = 8
                    codec = blosc2.Codec.ZSTD
                    cparams = {'codec': codec, 'clevel': clevel, 'nthreads': 8}
                    blosc2.asarray(
                        np.ascontiguousarray(arr),
                        urlpath=outpath,
                        chunks=(64, 64, 64),
                        blocks=(16, 16, 16),
                        cparams=cparams,
                        mode='w'
                        # mmap_mode='w+'
                    )
                    save_json(
                        {'annotation': anno, 'spacing': spacing},
                        outpath[:-5] + '.json',
                        sort_keys=False
                    )
                    del arr
                    shutil.rmtree(my_temp)

                except KeyboardInterrupt as e:
                    shutil.rmtree(my_temp)
                    raise e

                except Exception as e:
                    import IPython
                    IPython.embed()
                    print(e)
                    print("FAILED:", ds, run, outpath)
                    raise e

            # Process runs in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(process_run, run) for run in ds.runs]
                # Optionally, wait for all to finish and catch exceptions
                for future in concurrent.futures.as_completed(futures):
                    future.result()  # This will re-raise any exception

        return


def download_raw_dataset(
        whitelist: List[str],
        out_dir: str = "/media/isensee/data/kaggle_byu_additional_data/data/",
        num_threads: int = 3):
    not_found = set()
    tmp_dir = join(out_dir, 'temp')
    valid_idx = [i for i in range(len(whitelist)) if not whitelist[i].startswith('tomo_')]
    print(f'Whitelist contains {len(whitelist) - len(valid_idx)} cases starting with "tomo_". These are not yet on CZIportal!')
    whitelist = [whitelist[i] for i in valid_idx]

    maybe_mkdir_p(tmp_dir)
    datasets = set([i.split('__')[0] for i in whitelist])

    client = Client()

    for dataset in datasets:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            ds = Dataset.find(client, [Dataset.id == int(dataset)])
            assert len(ds) == 1
            ds = ds[0]
            valid_run_names = [i.split('__')[1] for i in whitelist if i.startswith(f"{dataset}__")]
            runs = [i for i in ds.runs if i.name in valid_run_names]
            found = [f"{dataset}__{r.name}" for r in runs]
            not_found_here = [i for i in whitelist if i.startswith(f"{dataset}__") and i not in found]
            if len(not_found_here) > 0:
                print(f'Could not find tomograms {not_found_here}!')
                not_found = not_found.union(not_found_here)

            print(f'Downloading {len(found)} runs for dataset {dataset} from authors {[i.name for i in ds.authors]}')

            for run in runs:
                futures.append(executor.submit(process_run, ds, run, out_dir, tmp_dir))
            for future in concurrent.futures.as_completed(futures):
                future.result()  # This will re-raise any exception

    return not_found


def process_run(dataset_here, run_here, out_dir, tmp_dir):
    outpath = os.path.join(out_dir, f"{dataset_here.id}__{run_here.name}.b2nd")
    if os.path.exists(outpath[:-5] + '.json'):
        print(f'skipping {run_here.name}')
        return

    my_temp = join(tmp_dir, run_here.name)
    if os.path.exists(my_temp):
        shutil.rmtree(my_temp)
    maybe_mkdir_p(my_temp)
    try:
        # Download tomo
        if len(run_here.tomograms) == 0:
            print(f'run {run_here.name} has no tomograms')
            return
        tomo = run_here.tomograms[0]
        tomo.download_omezarr(dest_path=my_temp)

        # Load tomo
        fpath = glob.glob(join(my_temp, "*"))[0]
        arr = zarr.open(fpath, mode='r')
        # import IPython;IPython.embed()
        spacing = arr.attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
        assert spacing[0] == tomo.voxel_spacing
        arr = arr['0']

        # Save
        clevel: int = 8
        codec = blosc2.Codec.ZSTD
        cparams = {'codec': codec, 'clevel': clevel, 'nthreads': 32}
        blosc2.asarray(
            np.ascontiguousarray(arr),
            urlpath=outpath,
            chunks=(64, 64, 64),
            blocks=(16, 16, 16),
            cparams=cparams,
            mode='w'
            # mmap_mode='w+'
        )
        save_json(
            {'spacing': spacing},
            outpath[:-5] + '.json',
            sort_keys=False
        )
        del arr
        shutil.rmtree(my_temp)

    except KeyboardInterrupt as e:
        shutil.rmtree(my_temp)
        raise e

    except Exception as e:
        import IPython
        IPython.embed()
        print(e)
        print("FAILED:", dataset_here.id, run_here.name, outpath)
        raise e


if __name__ == "__main__":
    wl = load_json('/home/isensee/drives/E132-Rohdaten/nnUNetv2/Dataset189_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_512/train_OrigShapes.json').keys()
    # wl = ['10165__ycw2012-02-25-18']
    # not_found = download_raw_dataset(
    #     whitelist=list(wl),
    #     out_dir='/media/isensee/T9/raw_data/kaggle_raw_data_download_dev',
    #     num_threads=4
    # )
    # print("Not found:", not_found)

