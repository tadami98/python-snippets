#!/usr/bin/env python3

import os
import sys
import subprocess
import requests
from getpass import getpass

GREEN = '\033[92m'
RED = '\033[91m'
ENDC = '\033[0m'

path = os.path.abspath(sys.argv[1])
repo_name = os.path.basename(path)

def print_info(message):
    print(f"{GREEN}[INFO]{ENDC} {message}")

def print_error(message):
    print(f"{RED}[ERROR]{ENDC} {message}", file=sys.stderr)

def create_local_repo(path):
    try:
        subprocess.run(['git', 'init'], cwd=path, check=True)
        subprocess.run(f"echo '# {repo_name}' > README.md", shell=True, check=True)
        subprocess.run(['git', 'add', '.'], cwd=path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=path, check=True)
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create local repository: {e}")
        sys.exit(1)

def create_github_repo(repo_name, token):
    url = 'https://api.github.com/user/repos'
    headers = {'Authorization': f'token {token}'}
    data = {'name': repo_name}

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 201:
        print_info(f"Repository '{repo_name}' created on GitHub.")
        return response.json()['ssh_url']
    else:
        print_error(f"Failed to create GitHub repository: {response.json()}")
        sys.exit(1)

def add_remote_and_push(path, repo_url):
    try:
        subprocess.run(['git', 'remote', 'add', 'origin', repo_url], cwd=path, check=True)
        subprocess.run(['git', 'branch', '-M', 'main'], cwd=path, check=True)
        subprocess.run(['git', 'push', '-u', 'origin', 'main'], cwd=path, check=True)
        print_info(f"Code pushed to GitHub repository.")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to push code to GitHub: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print_error("Usage: git-create <path>")
        sys.exit(1)

    if not os.path.isdir(path):
        print_error(f"The path '{path}' is not a valid directory.")
        sys.exit(1)

    print_info(f"Creating repository '{repo_name}'.")

    # Get GitHub token from environment or prompt user and create it
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        token = getpass("Enter your GitHub token: ")
        try:
            fish_config_path = os.path.expanduser("~/.config/fish/config.fish")
            with open(fish_config_path, "a") as config_file:
                config_file.write(f"\nset -x GITHUB_TOKEN \"{token}\"\n")
            print("Token added to config.fish successfully!")
        except FileNotFoundError:
            print("Error: config.fish file not found.")
        except Exception as e:
            print(f"Error writing to config.fish: {e}")

    create_local_repo(path)
    repo_url = create_github_repo(repo_name, token)
    add_remote_and_push(path, repo_url)

if __name__ == "__main__":
    main()
