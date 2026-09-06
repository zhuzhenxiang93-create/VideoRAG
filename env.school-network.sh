# University of Auckland Centre for eResearch documented proxy.
# https://uoa-eresearch.github.io/vmhandbook/doc/linux-proxy.html
# Opt in per project shell; no credentials and no system-wide settings.
export http_proxy=http://squid.auckland.ac.nz:3128
export https_proxy=http://squid.auckland.ac.nz:3128
export no_proxy=localhost,127.0.0.1,localaddress,.auckland.ac.nz,169.254.169.254
