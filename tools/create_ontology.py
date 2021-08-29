
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  group = parser.add_argument_group(title='input')
  group.add_argument('--lang', type=str, required=True,
                       help='the language being processed, e.g., pt, en, fr, es, ar, zh, hi, sw')
  group.add_argument('--shared_dir', type=str, default="/content/drive/Shareddrives/BigScience",
                       help='where to store the processed files')
  group.add_argument('--ken_lm_location', type=str, default="/content/drive/Shareddrives/BigScience/kenlm/bin/lmplz",
                       help='where to ken_lm binary is stored')
  args = parser.parse_args()
  for lang in args.lang.split():
    gather_ngram(arg, lang, force=False)

  
if False:
#if __name__ == "__main__":
  import datastore.distributed_context.DistributedContext
  from datastore.utils.utils import wait_until_files_loaded, get_oscar_urls, _download_urls
  os.environ['MEMORY'] = '20'
  parser = argparse.ArgumentParser()
  group = parser.add_argument_group(title='input')
  group.add_argument('--dask_scheduler_file', type=str, default="/content/drive/Shareddrives/BigScience/dask_scheduler.txt",
                       help='Path to dask scheduler file')
  group.add_argument('--ngrok_token_environ_var', type=str, default="NGROK_TOKEN",
                       help='Environmental variable containing ngrok token')
  group.add_argument('--shared_dir', type=str, default="/content/drive/Shareddrives/BigScience",
                       help='where to store the processed files')
  args = parser.parse_args()
  distributed_context = DistributedContext(dask_scheduler_file=args.dask_scheduler_file, ngrok_token=os.environ[args.ngrok_token_environ_var], launch_streamlit_app=False)
  results = ddistributed_context.dask_client.map(gather_ngram,  ("pt", "en", "fr", "es", "ar", "zh", "hi", "sw"))
  results.result()
  files = " ".join(wait_until_files_loaded(results.result()))
  os.system(f"gunzip {files} -c| {args.shared_dir}/terashuf > shuffled.txt")
  os.system(f"gzip shuffled.txt")
  os.system(f"mv shuffled.txt.gz {args.shared_dir}/")
  