import glob
import random
import os
import datetime


class FilesWriterBinaryUTF8:

    def __init__(self, basefilenamepath, chunk_size=int(10*1024**2), verbose=True):
        """
        10*1024**2 is M of disk space.
        The saved files will try to be approx just a bit more than that.
        """
        self.chunk_size = chunk_size
        self.counter = -1
        self.basefilenamepath = basefilenamepath
        self.f = None

    def __enter__(self):
        self._append_to_new_file()
        return self

    def __exit__(self, type, value, traceback):
        self.f.close()

    def write(self, string_):
        self.f.write(
            bytes(string_, 'utf-8')
        )
        if self._is_current_file_too_big():
            self._append_to_new_file()

    def _is_current_file_too_big(self):
        is_current_file_too_big = (os.stat(self._get_current_file_name()).st_size > self.chunk_size)
        return is_current_file_too_big

    def _append_to_new_file(self):
        self.counter += 1
        if self.f is not None:
            self.f.close()
        new_filename = self._get_current_file_name()
        new_dirname = os.path.dirname(new_filename)
        if not os.path.exists(new_dirname):
            os.mkdir(new_dirname)
        self.f = open(new_filename, 'wb')
        print(
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "- Writing to:",
            new_filename
        )
        return self.f

    def _get_current_file_name(self):
        new_filename = self.basefilenamepath + "." + str(self.counter).zfill(15) + ".txt"
        return new_filename


class FilesReaderBinaryUTF8:

    def __init__(self, basefilenamepath, pick_files_in_random_order=True, verbose=False, min_paragraph_char_len=32):
        self.basefilenamepath = basefilenamepath
        self.files = glob.glob(basefilenamepath + "*")
        self.pick_files_in_random_order = pick_files_in_random_order
        self.verbose = verbose
        self.min_paragraph_char_len = min_paragraph_char_len
        self.f = None

    def __enter__(self):
        self.generator = self._generator_function().__iter__()
        return self

    def _generator_function(self):
        while True:
            if self.pick_files_in_random_order:
                random.shuffle(self.files)

            # loop through each files, possibly in random order
            for filename in self.files:
                if self.verbose:
                    print(
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "- Reading from:",
                        filename
                    )
                with open(filename, 'rb') as f:
                    text_chunk = f.read().decode('utf-8')
                    # loaded the file.

                    # split on double-newlines as paragraphs:
                    for paragraph in text_chunk.split("\n\n"):
                        paragraph = paragraph.strip().strip("\n")

                        # return only paragraphs having at least 32 characters:
                        if len(paragraph) >= self.min_paragraph_char_len:
                            yield paragraph

    def __exit__(self, type, value, traceback):
        del self.generator

    def next_paragraph(self):
        return self.generator.__next__()
