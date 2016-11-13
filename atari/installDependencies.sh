mkdir tmp
echo "Installing Xitari ... "
cd /tmp
rm -rf xitari
git clone https://github.com/deepmind/xitari.git
cd xitari
luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Xitari installation completed"

echo "Installing Alewrap ... "
cd /tmp
rm -rf alewrap
git clone https://github.com/deepmind/alewrap.git
cd Alewrap
luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Alewrap installation completed"
