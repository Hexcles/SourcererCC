#!/bin/env python2
import logging
import ctypes
import multiprocessing
import re
import collections
import os
import sys
import shutil
import hashlib
import datetime as dt
import argparse
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser # ver. < 3.0

# OUTPUT FORMATS
# TOKEN files: one line per block
# parentID,blockID@#@[token@@::@@count,]*
# e.g.: 1,2@#@for@@::@@1,"Fileset@@::@@1,perform@@::@@2,was@@::@@1,configured"@@::@@1,throw@@::@@1
# ------------
# BOOKKEEPING files: one line per block
# blockID,filepath,line_start,line_end
# e.g.: 0,/home/shawnwang/CloneDetector.java,46,89
# ------------
# STAT files: one line per block (CSV format)
# parentID, blockID, file_path, file_hash, file_size, lines, LOC, SLOC, tokens_count_total, tokens_count_unique

argparser = argparse.ArgumentParser(description='File-based tokenizer for SourcererCC.')
argparser.add_argument('project', metavar='<PROJECT_LIST>', help='Path to a project list file (one path per line)')
argparser.add_argument('-c', dest='config', default='config.ini', help='Path to the config file (default=config.ini)')
argparser.add_argument('-f', dest='force', action='store_true', default=False, help='Overwrite existing results')

MULTIPLIER = 50000000

# Global configurations to be overriden by the config file.
# Main
N_PROCESSES = 2
PROJECTS_BATCH = 20
INTRA_PROJECT = True
# Misc
PATH_stats_file_folder = 'files_stats'
PATH_bookkeeping_proj_folder = 'bookkeeping_projs'
PATH_tokens_file_folder = 'files_tokens'
PATH_logs = 'logs'
# Language
separators = ''
comment_inline = ''
comment_inline_pattern = comment_inline + '.*?$'
comment_open_tag = ''
comment_close_tag = ''
comment_open_close_pattern = comment_open_tag + '.*?' + comment_close_tag
file_extensions = ['.none']
########################################################

# Global variables owned by each process, to be initialized by init_process.
G_block_counter = None
G_project_counter = None
G_stats_filename = None
G_bookkeeping_filename = None
G_tokens_filename = None


def read_config(config_file):
    config = ConfigParser()
    try:
        config.read(config_file)
    except IOError:
        sys.stderr.write("Unable to read config file: %s\n" % config_file)
        sys.exit(-1)

    conf = {}
    conf['N_PROCESSES'] = config.getint('Main', 'N_PROCESSES')
    conf['PROJECTS_BATCH'] = config.getint('Main', 'PROJECTS_BATCH')
    conf['INTRA_PROJECT'] = config.get('Main', 'INTRA_PROJECT').strip().upper() == 'ON'

    conf['PATH_stats_file_folder'] = config.get('Misc', 'PATH_stats_file_folder')
    conf['PATH_bookkeeping_proj_folder'] = config.get('Misc', 'PATH_bookkeeping_proj_folder')
    conf['PATH_tokens_file_folder'] = config.get('Misc', 'PATH_tokens_file_folder')
    conf['PATH_logs'] = config.get('Misc', 'PATH_logs')

    conf['separators'] = config.get('Language', 'separators').strip('"').split(' ')
    conf['comment_inline'] = re.escape(config.get('Language', 'comment_inline'))
    conf['comment_inline_pattern'] = conf['comment_inline'] + '.*?$'
    conf['comment_open_tag'] = re.escape(config.get('Language', 'comment_open_tag'))
    conf['comment_close_tag'] = re.escape(config.get('Language', 'comment_close_tag'))
    conf['comment_open_close_pattern'] = conf['comment_open_tag'] + '.*?' + conf['comment_close_tag']
    conf['file_extensions'] = config.get('Language', 'file_extensions').split(' ')

    return conf


def apply_config(conf):
    global N_PROCESSES, PROJECTS_BATCH, INTRA_PROJECT
    global PATH_stats_file_folder, PATH_bookkeeping_proj_folder, PATH_tokens_file_folder, PATH_logs
    global separators, comment_inline, comment_inline_pattern, comment_open_tag, comment_close_tag, comment_open_close_pattern, file_extensions

    N_PROCESSES = conf['N_PROCESSES']
    PROJECTS_BATCH = conf['PROJECTS_BATCH']
    INTRA_PROJECT = conf['INTRA_PROJECT']

    PATH_stats_file_folder = conf['PATH_stats_file_folder']
    PATH_bookkeeping_proj_folder = conf['PATH_bookkeeping_proj_folder']
    PATH_tokens_file_folder = conf['PATH_tokens_file_folder']
    PATH_logs = conf['PATH_logs']

    separators = conf['separators']
    comment_inline = conf['comment_inline']
    comment_inline_pattern = conf['comment_inline_pattern']
    comment_open_tag = conf['comment_open_tag']
    comment_close_tag = conf['comment_close_tag']
    comment_open_close_pattern = conf['comment_open_close_pattern']
    file_extensions = conf['file_extensions']


def init_process(conf, block_counter, project_counter):
    pid = os.getpid()
    apply_config(conf)

    # Shared objects inherited from the main process.
    global G_block_counter, G_project_counter
    G_block_counter = block_counter
    G_project_counter = project_counter

    # Names of output files of this process.
    global G_stats_filename, G_bookkeeping_filename, G_tokens_filename
    G_stats_filename = os.path.join(PATH_stats_file_folder, 'stats-%d.stats' % pid)
    G_bookkeeping_filename = os.path.join(PATH_bookkeeping_proj_folder, 'bookkeeping-%d.projs' % pid)
    G_tokens_filename = os.path.join(PATH_tokens_file_folder, 'tokens-%d.tokens' % pid)

    # Set up logging for this process.
    FORMAT = '[%(levelname)s](' + str(os.getpid()) + ') %(message)s'
    # A default stdio handler with INFO level.
    logging.basicConfig(level=logging.WARNING, format=FORMAT)
    # A file handler with DEBUG level.
    file_handler = logging.FileHandler(os.path.join(PATH_logs,'LOG-'+str(os.getpid())+'.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FORMAT))
    logging.getLogger().addHandler(file_handler)
    logging.info("Proccess %d started." % os.getpid())


# Returns:
# ((file_hash, lines, LOC, SLOC),
#  (tokens_count_total, tokens, count_unique, tokens_hash, tokens),
#  [separator_time, token_time, hash_time, re_time])
def tokenize_files(file_string, comment_inline_pattern, comment_open_close_pattern, separators):
    final_stats  = 'ERROR'
    final_tokens = 'ERROR'

    file_hash = 'ERROR'
    lines     = 'ERROR'
    LOC       = 'ERROR'
    SLOC      = 'ERROR'

    h_time = dt.datetime.now()
    m = hashlib.md5()
    m.update(file_string)
    file_hash = m.hexdigest()
    hash_time = dt.datetime.now() - h_time

    lines = file_string.count('\n')
    if not file_string.endswith('\n'):
        lines += 1
    file_string = "".join([s for s in file_string.splitlines(True) if s.strip()])

    LOC = file_string.count('\n')
    if not file_string.endswith('\n'):
        LOC += 1

    re_time = dt.datetime.now()
    # Remove tagged comments
    file_string = re.sub(comment_open_close_pattern, '', file_string, flags=re.DOTALL)
    # Remove end of line comments
    file_string = re.sub(comment_inline_pattern, '', file_string, flags=re.MULTILINE)
    re_time = dt.datetime.now() - re_time

    file_string = "".join([s for s in file_string.splitlines(True) if s.strip()]).strip()

    SLOC = file_string.count('\n')
    if file_string != '' and not file_string.endswith('\n'):
        SLOC += 1

    final_stats = (file_hash, lines, LOC, SLOC)

    # Rather a copy of the file string here for tokenization
    file_string_for_tokenization = file_string

    #Transform separators into spaces (remove them)
    s_time = dt.datetime.now()
    for x in separators:
        file_string_for_tokenization = file_string_for_tokenization.replace(x,' ')
    s_time = dt.datetime.now() - s_time

    ##Create a list of tokens
    file_string_for_tokenization = file_string_for_tokenization.split()
    ## Total number of tokens
    tokens_count_total = len(file_string_for_tokenization)
    ##Count occurrences
    file_string_for_tokenization = collections.Counter(file_string_for_tokenization)
    ##Converting Counter to dict because according to StackOverflow is better
    file_string_for_tokenization=dict(file_string_for_tokenization)
    ## Unique number of tokens
    tokens_count_unique = len(file_string_for_tokenization)

    t_time = dt.datetime.now()
    #SourcererCC formatting
    tokens = ','.join(['{}@@::@@{}'.format(k, v) for k,v in file_string_for_tokenization.iteritems()])
    t_time = dt.datetime.now() - t_time

    # MD5
    h_time = dt.datetime.now()
    m = hashlib.md5()
    m.update(tokens)
    hash_time += dt.datetime.now() - h_time

    final_tokens = (tokens_count_total, tokens_count_unique, m.hexdigest(), tokens)

    return (final_stats, final_tokens, [s_time, t_time, hash_time, re_time])


def process_file(file_string, file_size, file_path, project_id, file_id,
                 tokens_file, bookkeeping_file, stats_file):
    logging.info('Parsing file: ' + file_path)

    (final_stats, final_tokens, file_parsing_times) = tokenize_files(file_string, comment_inline_pattern, comment_open_close_pattern, separators)

    (file_hash,lines,LOC,SLOC) = final_stats
    (tokens_count_total,tokens_count_unique,token_hash,tokens) = final_tokens

    ww_time = dt.datetime.now()
    stats_file.write(','.join(
        [str(project_id), str(file_id), file_path, file_hash, str(file_size),
         str(lines), str(LOC), str(SLOC), str(tokens_count_total), str(tokens_count_unique)]
    ) + '\n')
    w_time = dt.datetime.now() - ww_time

    ww_time = dt.datetime.now()
    tokens_file.write('%d,%d@#@%s\n' % (project_id, file_id, tokens))
    w_time += dt.datetime.now() - ww_time

    ww_time = dt.datetime.now()
    bookkeeping_file.write('%d,%s,%d,%d\n' % (file_id, file_path, 1, lines))
    w_time += dt.datetime.now() - ww_time

    logging.info('Successfully parsed: ' + file_path)
    return [w_time] + file_parsing_times  # [w_time, s_time, t_time, hash_time, re_time]


def walk_directory(args, current_dir, filenames):
    project_id, project_path, tokens_file, bookkeeping_file, stats_file, durations = args

    for filename in filenames:
        file_path = os.path.join(current_dir, filename)
        if not os.path.isfile(file_path):
            continue
        # Test the file extension.
        _, ext = os.path.splitext(filename)
        if ext not in file_extensions:
            continue

        if project_id is None:
            # Intra-project ON. Unique project ID for each file.
            with G_project_counter.get_lock():
                parent_id = G_project_counter.value
                G_project_counter.value += 1
        else:
            # Intra-project OFF. project_id already provided by caller.
            parent_id = project_id
        with G_block_counter.get_lock():
            file_id = G_block_counter.value
            G_block_counter.value += 1

        file_size = os.path.getsize(file_path)
        logging.info(file_path)

        with open(file_path, 'r') as f:
            r_time = dt.datetime.now()
            file_content = f.read()
            r_time = dt.datetime.now() - r_time

            w_time, s_time, t_time, hash_time, re_time = \
                process_file(file_content, file_size, file_path, parent_id, file_id,
                             tokens_file, bookkeeping_file, stats_file)

            durations[0] += r_time.total_seconds()
            durations[1] += w_time.total_seconds()
            durations[2] += s_time.total_seconds()
            durations[3] += t_time.total_seconds()
            durations[4] += hash_time.total_seconds()
            durations[5] += re_time.total_seconds()


def process_project(project_path):
    p_start = dt.datetime.now()
    logging.info('Starting project: ' + project_path)

    if not os.path.isdir(project_path):
        logging.error('Unable to open project: ' + project_path)
        return 0

    if INTRA_PROJECT:
        project_id = None
    else:
        with G_project_counter.get_lock():
            project_id = G_project_counter.value
            G_project_counter.value += 1

    durations = [.0] * 6
    with open(G_tokens_filename, 'a+') as tokens_file, \
            open(G_bookkeeping_filename, 'a+') as bookkeeping_file, \
            open(G_stats_filename, 'a+') as stats_file:
        os.path.walk(project_path, walk_directory,
                     (project_id, project_path,
                      tokens_file, bookkeeping_file, stats_file,
                      durations))

    p_elapsed = dt.datetime.now() - p_start
    read_time, write_time, separators_time, tokens_time, hash_time, regex_time = durations
    logging.info('Project %s finished in %s', project_path, p_elapsed)
    logging.info('Read: %ss Write: %ss Separators: %ss Tokens: %ss Hash: %ss Regex: %ss',
                 read_time, write_time, separators_time, tokens_time, hash_time, regex_time)
    return 1


def main():
    args = argparser.parse_args()
    conf = read_config(args.config)
    apply_config(conf)

    if os.path.exists(PATH_stats_file_folder) or os.path.exists(PATH_bookkeeping_proj_folder) \
            or os.path.exists(PATH_tokens_file_folder) or os.path.exists(PATH_logs):
        if args.force:
            shutil.rmtree(PATH_stats_file_folder)
            shutil.rmtree(PATH_bookkeeping_proj_folder)
            shutil.rmtree(PATH_tokens_file_folder)
            shutil.rmtree(PATH_logs)
        else:
            sys.stderr.write("Output path(s) already existed. Run with '-f'?\n")
            sys.exit(-1)
    os.makedirs(PATH_stats_file_folder)
    os.makedirs(PATH_bookkeeping_proj_folder)
    os.makedirs(PATH_tokens_file_folder)
    os.makedirs(PATH_logs)

    p_start = dt.datetime.now()

    print("*** Reading project list...")
    project_paths = []
    with open(args.project, 'r') as f:
        for line in f:
            project_paths.append(line.strip())

    block_counter = multiprocessing.Value(ctypes.c_uint64, 0)
    project_counter = multiprocessing.Value(ctypes.c_uint64, 0)
    # initializer and initargs are specified because on Windows we can't rely
    # on the behaviour of fork() to share global variables.
    process_pool = multiprocessing.Pool(N_PROCESSES, init_process, [conf, block_counter, project_counter])

    print("*** Starting regular projects...")
    project_count = 0
    result_iter = process_pool.imap_unordered(process_project, project_paths, PROJECTS_BATCH)
    for r in result_iter:
        project_count += r

    p_elapsed = dt.datetime.now() - p_start
    print("*** All done. %d/%d projects successfully processed within %s." %
          (project_count, len(project_paths), p_elapsed))


if __name__ == '__main__':
    # To support py2exe and alike (though we aren't using them at the moment.)
    multiprocessing.freeze_support()
    main()
