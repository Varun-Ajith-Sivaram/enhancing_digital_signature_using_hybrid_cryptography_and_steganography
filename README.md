# enhancing_digital_signature_using_hybrid_cryptography_and_steganoraphy

Developed a Python application to improve the security of digital siganture of an image using hybrid cryptography and image steganography. 

The image is encrypted using AES algorithm. Different modes of AES are considered and encryption is performed using a symmetric key and a random AES mode selected at runtime. The symmetric key is encrypted and signed using RSA algorithm. The digital signature of the image is embedded into a cover image using LSB image steganography approach. Unlike in normal LSB method, prime numbered pixel positions are considered for embedding the digital signature in this approach as it scatters the data throughout the cover image hence, improving the security.
