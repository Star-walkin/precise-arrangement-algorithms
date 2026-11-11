from html.parser import HTMLParser
from common20 import *
links=[]
class Check(HTMLParser):
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src','poster') and not v.startswith('#'):
                assert (ROOT/v).exists(),v
                links.append(v)
Check().feed((ROOT/'report.html').read_text(encoding='utf8'))
release=read(ROOT/'release_manifest.json')
assert all(sha(ROOT/k)==v for k,v in release['receipts'].items())
dump(ROOT/'report_link_audit.json',dict(passed=True,local_links=len(links),release_receipt_hashes_verified=True))
print('Verified local report links',len(links),'and release receipt hashes')
