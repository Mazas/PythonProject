from urllib.parse import urljoin
import os
from requests import get
from bs4 import BeautifulSoup as soup


def get_page(base_url):
    req = get(base_url)
    if req.status_code == 200:
        return req.text
    raise Exception('Error {0}'.format(req.status_code))


def get_all_links(html):
    features = "html.parser"
    bs = soup(html, features)
    links = bs.findAll('a')
    return links


def get_zip(base_url, file_dir):
    html = get_page(base_url)
    links = get_all_links(html)
    if len(links) == 0:
        raise Exception('No links found on the web-page')
    n_zips = 0
    for link in links:
        if link['href'][-4:] == '.zip':
            n_zips += 1
            content = get(urljoin(base_url, link['href']))
            if content.status_code == 200 and content.headers['content-type'] == 'application/zip':
                with open(os.path.join(file_dir, link.text), 'wb') as zip_file:
                    print("Writing: {0}".format(link.text))
                    zip_file.write(content.content)
    if n_zips == 0:
        raise Exception('No zips found on the page')
    print("{0} zips downloaded and saved in {1}".format(n_zips, file_dir))


if __name__ == '__main__':
    dir_name = "2020-04-2"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print("Directory ", dir_name, " Created ")
    else:
        print("Directory ", dir_name, " already exists")

    url = "http://www.mordrek.com/goblinSpy/Replays/{0}/".format(dir_name)
    base_dir = "{0}\\{1}".format(os.getcwd(), dir_name)
    try:
        get_zip(url, base_dir)
    except Exception as e:
        print(e)
        exit(-1)
