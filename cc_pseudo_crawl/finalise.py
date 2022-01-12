from argparse import ArgumentParser

from datasets import load_dataset, concatenate_datasets


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")

    args = parser.parse_args()

    return args

def compute_external_ids(ds):
    """This is done at the end of processing and we basically convert `external_urls` in `external_ids`"""
    # For each url, find the most recent row id corresponding to that url
    # All of the duplicate of a `url` are either all in that dictionary or not in that dictionary
    # This table allows me to do a double join so I can easily compute the ids.
    # We'll then go through that csv and add the ids to final dataset.
    # No duplicates guaranteed
    url_to_id_and_timestamp = {}
    # TODO: batch this
    for data in ds:
        url = data["url"]
        id_ = data["id"]
        timestamp = data["fetch_time"]
        if url in url_to_id_and_timestamp:
            old_id, old_time_stamp = url_to_id_and_timestamp[url]
            new_timestamp, new_id = max((timestamp, id_), (old_time_stamp, old_id))
            url_to_id_and_timestamp[url] = (new_id, new_timestamp)
        else:
            url_to_id_and_timestamp[url] = (id_, timestamp)

    # TODO: batch this
    for data in ds:
        # Not all urls are part of our index. We keep `external_urls` for this sake.
        data["external_ids"] = [
            url_to_id_and_timestamp[external_url][0]
            for external_url in data["external_urls"]
            if external_url in url_to_id_and_timestamp
        ]

    # with open(csv_output_dir / f"previous_to_next.csv", "w") as fo:
    #     writer = csv.writer(fo)
    #     writer.writerow(['previous_id', 'previous_url', 'next_id', 'next_url'])
    #     for data in ds:
    #         previous_id = data["id"]
    #         previous_url = data["url"]
    #         for external_url in data["external_urls"]:
    #             next_id = url_to_id_and_timestamp.get(external_url, None)[0]
    #             writer.writerow([previous_id, previous_url, next_id, external_url])

def assign_id(batch, indices):
    batch["id"] = indices
    return batch

def main():
    args = get_args()

    dataset_dict = load_dataset(args.dataset, use_auth_token=True)

    assert "train" not in dataset_dict

    # Concatenate all the splits together
    ds = concatenate_datasets(list(dataset_dict.values()))

    # Generate id
    ds = ds.map(assign_id, batched=True, with_indices=True)

    # Generate external_ids
    ds = compute_external_ids(ds)

    # Add as train split
    dataset_dict["train"] = ds
    dataset_dict.push_to_hub(args.dataset, private=True)
    pass

if __name__ == "__main__":
    main()