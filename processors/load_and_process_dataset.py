
def _get_oscar_urls(language, shuffled="unshuffled", deduplicated="deduplicated"):
  _BASE_DATA_URL_FORMAT_STR = ("https://s3.amazonaws.com/datasets.huggingface.co/oscar/1.0/{shuffled}/{deduplicated}/{language}/")
  _BASE_CHECKSUM_FILE_NAME = "{language}_sha256.txt"
  base_data_url = _BASE_DATA_URL_FORMAT_STR.format(
            shuffled=shuffled, language=language, deduplicated=deduplicated
        )
  checksum_url = base_data_url + _BASE_CHECKSUM_FILE_NAME.format(language=language)
  with fsspec.open(checksum_url, encoding="utf-8") as f:
    data_filenames = [line.decode().split("\t")[0] for line in f if line]
    return [base_data_url + data_filename for data_filename in data_filenames]

def _download_urls(urls):
  for url in urls:
    if not os.path.exists(url.split("/")[-1]):
      os.system(f"wget {url}")

class MT5Processor:
  def init(self):
    self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
  
  def process(self, line)
    return " ".join(self.tokenizer.tokenize(line)).replace(mt5_underscore+" ", mt5_underscore)
          
class DatasetLoader:
  ### Load and parse initial subset of data
  """ Load an oscar subset for all langs except swahili, and flax-community/swahili-safi for swahili """
  

  def __init__(self, cjk_processor, processor=None):
    if cjk_processor is None:
      cjk_processor = MT5Processor()
    if processor is None:
      processor = MT5Processor()
    self.cjk_processor = cjk_processor
    self.processor = processor

  @staticmethod
  def parse_tok(file, processor):
    processor.init()
    with open(file.replace(".gz", ""), "w", encoding="utf8") as f:
      with gzip.open(file, "rb") as f2:
        for line in f2:
          line = line.decode().strip()
          if line:
            line = processor.process(line)
            f.write(line+"\n")
    #os.unlink(file)

  @staticmethod
  def parse_swahili_safi(batch, indices, processor):
    processor.init()
    shard = indices[0]
    with open(f"sw_{shard}.txt", "w", encoding="utf8") as f:
      for i in range(len(batch['text'])):
        line = processor.process(batch['text'][i])
        f.write(line+"\n")

  def tokenize_data_subset(self, arg, lang,  force=True, processor=None):
    shared_dir = arg.shared_dir
    if lang != "sw":
      if force or not os.path.exists(f"{shared_dir}/{lang}.txt.gz"):
        lst = _get_oscar_urls(lang)
        if len(lst) > 5:
          lst = sample(lst,5)
        _download_urls(lst)
        print (glob.glob(f"{lang}_*.txt.gz"))
        processes = [multiprocessing.Process(target=DatasetLoader.parse_tok, args=(file,self.cjk_processor if lang in ("zh", "ja", "ko") else self.processor)) for file in glob.glob(f"{lang}_*.txt.gz")]
        for process in processes:
          process.start()
        for process in processes:
          process.join()
        os.system(f"cat {lang}_*.txt > {lang}.txt")
        os.system(f"rm {lang}_*.txt {lang}_*.txt.gz")
        os.system(f"gzip {lang}.txt")
        os.system(f"cp {lang}.txt.gz {shared_dir}")
      return f"{shared_dir}/{lang}.txt.gz"
    else:
      if force or not os.path.exists(f"{shared_dir}/sw.txt.gz"):
        from datasets import load_dataset
        ds = load_dataset("flax-community/swahili-safi")
        ds['train'].map(DatasetLoader.parse_swahili_safi, batched=True, batch_size=int(len(ds['train'])/4), num_proc=4, with_indices=True, fn_kwargs={'processor': self.processor})
        os.system(f"cat {lang}_*.txt > {lang}.txt")
        os.system(f"rm {lang}_*.txt {lang}_*.txt.gz")
        os.system(f"gzip {lang}.txt")
        os.system(f"cp {lang}.txt.gz {shared_dir}")
      return f"{shared_dir}/{lang}.txt.gz"