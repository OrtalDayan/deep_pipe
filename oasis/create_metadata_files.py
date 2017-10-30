import os
import fnmatch
import nibabel as nib
import pandas as pd


class PathHelper(object):
    def __init__(self, parent_path, data_dir_name):
        self.parent_path = parent_path
        self.data_dir_name = data_dir_name

    def record_path(self, disc, record):
        return os.path.join(self.data_path(), disc, record)

    def disc_path(self, disc):
        return os.path.join(self.data_path(), disc)

    def data_path(self):
        return os.path.join(self.parent_path, self.data_dir_name)

    def path_for_csv(self, disc, record, file):
        return os.path.join(self.data_dir_name, disc, record, file)


def get_rel_image_path(path, directory, pattern) -> str:
    for file in os.listdir(os.path.join(path, directory)):
        if fnmatch.fnmatch(file, pattern):
            return os.path.join(directory, file)


def process_record(record, disc, path_helper):
    record_path = path_helper.record_path(disc, record)

    s_rel_path = get_rel_image_path(record_path, "PROCESSED/MPRAGE/T88_111", "{}_mpr_n*_111_t88_gfc.hdr".format(record))
    nib.load(os.path.join(record_path, s_rel_path))  # Checking validity
    s_csv_path = path_helper.path_for_csv(disc, record, s_rel_path)

    t_rel_path = get_rel_image_path(record_path, "FSL_SEG", "{}_mpr_n*_anon_111_t88_masked_gfc_fseg.hdr".format(record))
    nib.load(os.path.join(record_path, t_rel_path))   # Checking validity
    t_csv_path = path_helper.path_for_csv(disc, record, t_rel_path)

    return s_csv_path, t_csv_path


def process_disc_dir(disc, path_helper, records) -> None:
    for record in os.listdir(path_helper.disc_path(disc)):
        if record == ".DS_Store":
            continue
        if fnmatch.fnmatch(record, "OAS*"):
            s_path, t_path = process_record(record, disc, path_helper)
            records[record] = {'S': s_path, 'T': t_path}
        else:
            raise RuntimeError


def create_metadata_csv(parent_dir_path, data_dir_name="data"):
    records = {}
    path_helper = PathHelper(parent_dir_path, data_dir_name)
    for disc in os.listdir(path_helper.data_path()):
        if disc == ".DS_Store":
            continue
        if not fnmatch.fnmatch(disc, "disc*"):
            raise RuntimeError
        process_disc_dir(disc, path_helper, records)

    metadata = pd.DataFrame.from_dict(records, 'index')
    metadata.sort_index(inplace=True)
    return metadata


if __name__ == '__main__':
    parent_dir = "/Users/gleb/Desktop/oasis"
    metadata = create_metadata_csv(parent_dir)
    metadata.to_csv(os.path.join(parent_dir, 'metadata.csv'), index_label='id')
