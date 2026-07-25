# install with proxy

## miniconda

curl -x http://127.0.0.1:31181 -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && echo "Downloaded OK" || echo "Failed"

chmod +x /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p ~/miniconda3 && echo "Install OK"

~/miniconda3/bin/conda init bash && echo "Init OK"



## config proxy temp
~/miniconda3/bin/conda config --set proxy_servers.http http://127.0.0.1:31181
~/miniconda3/bin/conda config --set proxy_servers.https http://127.0.0.1:31181
~/miniconda3/bin/conda config --show proxy_servers

## create

~/miniconda3/bin/conda create -n tomo-test python=3.10 numpy scipy matplotlib scikit-image pydicom -y 2>&1


~/miniconda3/bin/conda create -n tomo-test python=3.10 numpy scipy matplotlib scikit-image pydicom -y 2>&1 | tail -30
