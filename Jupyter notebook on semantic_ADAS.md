# Using Jupyter notebook and lab on semantic_ADAS using chrome

1. Use the drun command along with port mapping: 

   ```bash
   drun -a SOCISP -i Semantic_ADAS -c bash -q gpu_debug_q -I -v /opt/dockermounts/algo-datasets:/mnt/datasets -p 8123:8181 --pull
   ```

2. Open Jupyter notebook along with the correct mappings:

    ```bash
   /etc/anaconda3/bin/jupyter notebook --ip=0.0.0.0 --port=8181
    ```

3. Open Jupyter GUI from browser on windows using the gpu address (in case 8123 is the port the GPU is set to export, and the name of the GPU is gpu04-dt):

    ```html
   http://gpu04-dt:8123
    ```

4. You may be prompted with a token request, you will find it in your container bash after the jupyter notebook command in the following form:

   ```bash
   Copy/paste this URL into your browser when you connect for the first time,
       to login with a token:
           http://0.0.0.0:8181/?token=b0ff8c631b0e3521180a5ebed7271c5a7a83c4962ff9c1da
   
   ```

   take the part after 'token=', in this case here it is 

   ```bash
   b0ff8c631b0e3521180a5ebed7271c5a7a83c4962ff9c1da
   ```

   