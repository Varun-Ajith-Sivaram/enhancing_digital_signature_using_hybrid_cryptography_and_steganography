import cv2
from PIL import Image
import sys
import numpy as np
import random
import rsa
import hmac
import base64
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
from stegano import lsbset
from stegano.lsbset import generators
import skimage
from matplotlib import pyplot as plt
import os


def show_image(title, obj):
    cv2.imshow(title, obj)
    cv2.waitKey()
    cv2.destroyAllWindows()


def choose_img_file(img_dict):
    while True:
        print("\nAvailable images for simulation:")
        for i in img_dict:
            print(f"{i:5}.{img_dict[i]}")

        file = int(input("\n\tEnter the option:"))
        match file:
            case 1:
                file = img_dict.get(file)
                break
            case 2:
                file = img_dict.get(file)
                break
            case 3:
                file = img_dict.get(file)
                break
            case 4:
                file = img_dict.get(file)
                break
            case 5:
                file = img_dict.get(file)
                break
            case 6:
                file = img_dict.get(file)
                break
            case _:
                print("\n\t---> Invalid option, try again!!")
    return file


def choose_cover_img_file(cover_dict):
    while True:
        print("\nAvailable cover images for simulation:")
        for i in cover_dict:
            print(f"{i:5}.{cover_dict[i]}")

        file = int(input("\n\tEnter the option:"))
        match file:
            case 1:
                file = cover_dict.get(file)
                break
            case 2:
                file = cover_dict.get(file)
                break
            case 3:
                file = cover_dict.get(file)
                break
            case 4:
                file = cover_dict.get(file)
                break
            case _:
                print("\n\t---> Invalid option, try again!!")
    return file


def get_image_data(cv, pl):
    data = {
        "Image Width": pl.width,
        "Image Height": pl.height,
        "image Size": pl.size,
        "Image Format": '.' + pl.format.lower(),
        "Image Mode": pl.mode,
        "No. of channels": cv.shape[2]
    }

    return data


def get_cover_image_data(cv, pl):
    data = {
        "Image Width": pl.width,
        "Image Height": pl.height,
        "image Size": pl.size,
        "Image Format": '.' + pl.format.lower(),
        "Image Mode": pl.mode,
        "No. of channels": cv.shape[2]
    }

    return data


def draw_histogram(img1,img2):
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(img1[:, :, channel_id], bins=256, range=(0, 256))
        plt.plot(bin_edges[0:-1], histogram, color=c)
    plt.title("Cover Image"), plt.xlabel("Color value"), plt.ylabel("Pixel count"), plt.xlim([0, 256])

    plt.subplot(122)
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(img2[:, :, channel_id], bins=256, range=(0, 256))
        plt.plot(bin_edges[0:-1], histogram, color=c)
    plt.title("Stego Image"), plt.xlabel("Color value")
    plt.ylabel("Pixel count")
    plt.xlim([0, 256])

    plt.tight_layout()
    # plt.savefig(f"./images/histograms/{i}.png")
    plt.show()


def calculate_image_quality(img1, img2):
    psnr = skimage.metrics.peak_signal_noise_ratio(img1, img2)
    mse = skimage.metrics.mean_squared_error(img1, img2)
    ssim = skimage.metrics.structural_similarity(img1, img2, channel_axis=-1)

    print(f"Peak Signal to Noise Ratio(PSNR)= {psnr} dB")
    print(f"Mean Squared Error(MSE)= {mse} ")
    print(f"Structural Similarity(SSIM)= {ssim}\n")


if __name__ == "__main__":

    available_img = {
        1: "bird.jpg",
        2: "maple.jpg",
        3: "sea.jpg",
        4: "shapes.png",
        5: "sky.jpg",
        6: "tank.jpg"
    }

    available_cover_img = {
        1: "autumn.png",
        2: "desert.png",
        3: "island.png",
        4: "tree.png"
    }

    img_path = "C:\\Users\\Varun\\PycharmProjects\\pythonProject1\\images\\"
    cover_img_path = "C:\\Users\\Varun\\PycharmProjects\\pythonProject1\\images\\cover_image\\"

    file_name = choose_img_file(available_img)

    img = cv2.imread(img_path + file_name)
    if img is None:
        sys.exit("\n--> Error in loading image!!")
    else:
        print("\nImage successfully loaded...\n")

    show_image("Sender Side Image", img)

    img_pil = Image.open(img_path + file_name)
    img_info = get_image_data(img, img_pil)
    img_pil.close()

    print("\nInput Image Details:")
    for x in img_info:
        print(f"{x:17}: {img_info[x]}")

    print("\nSENDER(A) SIDE")
    print("--------------")

    success, img_encode = cv2.imencode(img_info["Image Format"], img)
    if success is None:
        sys.exit("\n--> Error in encoding image!!")
    else:
        print("\nEncoding image...")

    shape_img = img_encode.shape
    byte_string = img_encode.tobytes()
    print("Converting image data to bytes...")

    # key generation for AES encryption
    key = get_random_bytes(32)
    print("\nGenerated symmetric key for AES...")
    print("AES key: ", key)

    # AES encryption
    mode_list = [AES.MODE_CBC, AES.MODE_CTR, AES.MODE_OFB, AES.MODE_CFB]
    random.shuffle(mode_list)

    ch = random.choice(mode_list)
    aes_mode = ch

    global ct, iv, pt, nonce

    if aes_mode is AES.MODE_CBC:
        cipher = AES.new(key, aes_mode)
        ct_bytes = cipher.encrypt(pad(byte_string, AES.block_size))
        iv = base64.b64encode(cipher.iv).decode('utf-8')
        ct = base64.b64encode(ct_bytes).decode('utf-8')
        print("\nMode of AES chosen: CBC\n")
    elif aes_mode is AES.MODE_CTR:
        cipher = AES.new(key, aes_mode)
        ct_bytes = cipher.encrypt(byte_string)
        nonce = base64.b64encode(cipher.nonce).decode('utf-8')
        ct = base64.b64encode(ct_bytes).decode('utf-8')
        print("\nMode of AES chosen: CTR\n")
    elif aes_mode is AES.MODE_OFB:
        cipher = AES.new(key, aes_mode)
        ct_bytes = cipher.encrypt(byte_string)
        iv = base64.b64encode(cipher.iv).decode('utf-8')
        ct = base64.b64encode(ct_bytes).decode('utf-8')
        print("\nMode of AES chosen: OFB\n")
    else:
        cipher = AES.new(key, aes_mode)
        ct_bytes = cipher.encrypt(byte_string)
        iv = base64.b64encode(cipher.iv).decode('utf-8')
        ct = base64.b64encode(ct_bytes).decode('utf-8')
        print("\nMode of AES chosen: CFB\n")
    print("Encrypting byte string...")

    # HMAC generation
    hash_ct = hashlib.sha3_512(ct.encode()).hexdigest()
    print("\nGenerated digital signature...")
    print("Digital Signature: ", hash_ct)

    # key pair generation for AES key encryption using RSA
    (pub_keyB, prv_keyB) = rsa.newkeys(1024)
    print("\nGenerated key pair of recipient(B) for RSA...")
    (pub_keyA, prv_keyA) = rsa.newkeys(1024)
    print("Generated key pair of sender(A) for RSA...")
    encrypt_key = rsa.encrypt(key, pub_keyB)
    print("\nEncrypted symmetric key...")

    hash_algo = 'SHA-512'
    sign_key = rsa.sign(key, prv_keyA, hash_algo)
    print("Signed the symmetric key...")

    cover_file_name = choose_cover_img_file(available_cover_img)
    cover_img = cv2.imread(cover_img_path + cover_file_name)
    if cover_img is None:
        sys.exit("\n--> Error in loading image!!")
    else:
        print("\nImage successfully loaded...\n")

    show_image("Cover Image", cover_img)

    cover_img_pil = Image.open(cover_img_path + cover_file_name)
    cover_img_info = get_cover_image_data(cover_img, cover_img_pil)
    cover_img_pil.close()

    print("\nCover Image Details:")
    for x in cover_img_info:
        print(f"{x:17}: {cover_img_info[x]}")

    print("\nEmbedding digital signature to cover image...")
    secret = lsbset.hide(cover_img_path + cover_file_name, hash_ct, generators.eratosthenes())
    print("Stego image obtained...")
    secret.save("./stego_image.png")

    stego_img = cv2.imread("./stego_image.png")
    if stego_img is None:
        sys.exit("\n--> Error in loading image!!")
    else:
        print("\nImage successfully loaded...\n")

    show_image("Stego Image", stego_img)

    print("Generating histograms of Cover Image & Stego Image...")
    draw_histogram(cover_img, stego_img)

    print("\nCalculating Image Quality Indexes of Cover Image & Stego Image:")
    calculate_image_quality(cover_img, stego_img)

    print("\nRECIPIENT(B) SIDE")
    print("--------------")

    verified_algo = rsa.verify(key, sign_key, pub_keyA)
    if verified_algo == hash_algo:
        print("\nSymmetric key verification successful...")
    else:
        sys.exit("\n--> Program Aborted!!")

    key = rsa.decrypt(encrypt_key, prv_keyB)
    print("Decrypted symmetric key...")

    rec_ct = ct
    print("\nGenerating Digital Signature from the received Cipher Text...")
    hash_rec_ct = hashlib.sha3_512(rec_ct.encode()).hexdigest()
    print("Digital Signature from received cipher text: ")
    print("\t" + hash_rec_ct)

    print("\nExtracting Digital Signature embedded in Stego Image...")
    extract_hash = lsbset.reveal("./stego_image.png", generators.eratosthenes())
    print("Digital Signature from received Stego Image: ")
    print("\t" + extract_hash)

    try:
        os.remove("./stego_image.png")
    except OSError:
        pass

    status = hmac.compare_digest(hash_rec_ct, extract_hash)
    if status is True:
        print("\nDigital signature matches, proceeding to decryption...")
    else:
        sys.exit("\n--> Digital signature does not match/Stego Image corrupted, Program Aborted!!")

    if aes_mode is AES.MODE_CBC:
        rec_iv = base64.b64decode(iv)
        rec_ct = base64.b64decode(ct)
        rec_cipher = AES.new(key, aes_mode, rec_iv)
        pt = unpad(rec_cipher.decrypt(rec_ct), AES.block_size)
    elif aes_mode is AES.MODE_CTR:
        nonce = base64.b64decode(nonce)
        ct = base64.b64decode(ct)
        cipher = AES.new(key, aes_mode, nonce=nonce)
        pt = cipher.decrypt(ct)
    elif aes_mode is AES.MODE_OFB:
        iv = base64.b64decode(iv)
        ct = base64.b64decode(ct)
        cipher = AES.new(key, aes_mode, iv=iv)
        pt = cipher.decrypt(ct)
    else:
        iv = base64.b64decode(iv)
        ct = base64.b64decode(ct)
        cipher = AES.new(key, aes_mode, iv=iv)
        pt = cipher.decrypt(ct)

    print("\nDecrypted to byte string...")

    deserialized_bytes = np.frombuffer(pt, dtype=np.uint8)
    print("Converting byte string to image data...")
    deserialized_img = np.reshape(deserialized_bytes, shape_img)
    status = np.array_equal(img_encode, deserialized_img)
    if status is False:
        sys.exit("\n--> Deserialization Failed..")

    rec_img = cv2.imdecode(deserialized_img, cv2.IMREAD_COLOR)
    print("Decoding image data...")

    show_image("Recipient Side Image", rec_img)

    """
    print("\nGenerating histograms of Input Image & Output Image...")
    draw_histogram(img,rec_img)

    print("\nCalculating Image Quality Indexes of Cover Image & Stego Image:")
    calculate_image_quality(img, rec_img)
    """
