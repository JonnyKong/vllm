# SPDX-License-Identifier: Apache-2.0
import json


def main():
    profile1 = []
    profile2 = []
    with open("profile.json") as f:
        profile1 = json.load(f)

    with open("profile_second_iter.json") as f:
        profile2 = json.load(f)

    profile_merged = profile1 + profile2

    with open("profile_decode_only_uniform.json", "w") as f:
        json.dump(profile_merged, f, indent=4)


if __name__ == '__main__':
    main()
