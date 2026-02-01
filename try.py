# (.venv) PS C:\Code\python\vscode\music_genre_classifier> & C:\Code\python\vscode\music_genre_classifier\.venv\Scripts\python.exe c:/Code/python/vscode/music_genre_classifier/src/train.py 
# Using 8 dataloader workers
# [E0] epoch=01 loss=2.0030 train_acc=0.1801 val_acc=0.3162 time=378.2s
# [E0] epoch=02 loss=1.9015 train_acc=0.2412 val_acc=0.3925 time=315.1s
# [E0] epoch=03 loss=1.8416 train_acc=0.2648 val_acc=0.4300 time=265.5s
# [E0] epoch=04 loss=1.8010 train_acc=0.2773 val_acc=0.4213 time=269.4s
# [E0] epoch=05 loss=1.7802 train_acc=0.2849 val_acc=0.4537 time=279.4s
# [E0] epoch=06 loss=1.7488 train_acc=0.2844 val_acc=0.4662 time=275.6s
# [E0] epoch=07 loss=1.7278 train_acc=0.2816 val_acc=0.4587 time=274.3s
# [E0] epoch=08 loss=1.7206 train_acc=0.3032 val_acc=0.4813 time=273.6s
# [E0] epoch=09 loss=1.7048 train_acc=0.2873 val_acc=0.4387 time=274.6s
# [E0] epoch=10 loss=1.6966 train_acc=0.2967 val_acc=0.4850 time=268.6s
# [E0] epoch=11 loss=1.6868 train_acc=0.3172 val_acc=0.4562 time=268.7s
# [E0] epoch=12 loss=1.6831 train_acc=0.3128 val_acc=0.4925 time=269.1s
# [E0] epoch=13 loss=1.6606 train_acc=0.3048 val_acc=0.5075 time=266.3s
# [E0] epoch=14 loss=1.6671 train_acc=0.3098 val_acc=0.5275 time=265.5s
# [E0] epoch=15 loss=1.6631 train_acc=0.3117 val_acc=0.4925 time=265.7s
# [E0] epoch=16 loss=1.6543 train_acc=0.3108 val_acc=0.5000 time=265.0s
# [E0] epoch=17 loss=1.6375 train_acc=0.3200 val_acc=0.4600 time=263.1s
# [E0] epoch=18 loss=1.6485 train_acc=0.3137 val_acc=0.5387 time=262.9s
# [E0] epoch=19 loss=1.6447 train_acc=0.3100 val_acc=0.5238 time=262.2s
# [E0] epoch=20 loss=1.6319 train_acc=0.2921 val_acc=0.4700 time=261.6s
# [E0] epoch=21 loss=1.6306 train_acc=0.3265 val_acc=0.5337 time=262.0s
# [E0] epoch=22 loss=1.6275 train_acc=0.3320 val_acc=0.5012 time=263.0s
# [E0] epoch=23 loss=1.6052 train_acc=0.3337 val_acc=0.5625 time=260.5s
# [E0] epoch=24 loss=1.6198 train_acc=0.3387 val_acc=0.5288 time=259.4s
# [E0] epoch=25 loss=1.6136 train_acc=0.3209 val_acc=0.5275 time=263.4s
# [E0] epoch=26 loss=1.6085 train_acc=0.3201 val_acc=0.5275 time=261.2s
# [E0] epoch=27 loss=1.6224 train_acc=0.3173 val_acc=0.5387 time=259.8s
# [E0] epoch=28 loss=1.6135 train_acc=0.3217 val_acc=0.5225 time=258.5s
# [E0] epoch=29 loss=1.6023 train_acc=0.3232 val_acc=0.5563 time=262.1s
# [E0] epoch=30 loss=1.6042 train_acc=0.3269 val_acc=0.5162 time=259.8s
# [E0] epoch=31 loss=1.6031 train_acc=0.3303 val_acc=0.4863 time=257.8s
# Early stop at epoch 31, best_val_acc=0.5625
# Traceback (most recent call last):
#   File "c:\Code\python\vscode\music_genre_classifier\src\train.py", line 353, in <module>
#     main()
#   File "c:\Code\python\vscode\music_genre_classifier\src\train.py", line 333, in main
#     save_curves(log_csv, curve_png)
#   File "c:\Code\python\vscode\music_genre_classifier\src\train.py", line 100, in save_curves
#     e, tl, va, sec = line.strip().split(",")
#     ^^^^^^^^^^^^^^
# ValueError: too many values to unpack (expected 4)

# [E0] epoch=01 loss=2.0063 train_clip_acc=0.1875 train_song_acc=0.1896 val_acc=0.3375 time=439.4s
# [E0] epoch=02 loss=1.8993 train_clip_acc=0.2460 train_song_acc=0.2534 val_acc=0.4037 time=362.7s
# [E0] epoch=03 loss=1.8418 train_clip_acc=0.2634 train_song_acc=0.2758 val_acc=0.4325 time=296.8s
# [E0] epoch=04 loss=1.8016 train_clip_acc=0.2794 train_song_acc=0.2954 val_acc=0.4525 time=314.3s
# [E0] epoch=05 loss=1.7755 train_clip_acc=0.2919 train_song_acc=0.3012 val_acc=0.4425 time=308.3s
# [E0] epoch=06 loss=1.7413 train_clip_acc=0.2886 train_song_acc=0.3040 val_acc=0.4600 time=308.1s
# [E0] epoch=07 loss=1.7176 train_clip_acc=0.2887 train_song_acc=0.3051 val_acc=0.4562 time=306.8s
# [E0] epoch=08 loss=1.7171 train_clip_acc=0.3073 train_song_acc=0.3250 val_acc=0.5062 time=301.6s
# [E0] epoch=09 loss=1.7013 train_clip_acc=0.2966 train_song_acc=0.3111 val_acc=0.4825 time=299.6s
# [E0] epoch=10 loss=1.6957 train_clip_acc=0.2985 train_song_acc=0.3140 val_acc=0.4850 time=300.6s
# [E0] epoch=11 loss=1.6968 train_clip_acc=0.3044 train_song_acc=0.3197 val_acc=0.4325 time=305.0s
# [E0] epoch=12 loss=1.6817 train_clip_acc=0.3043 train_song_acc=0.3178 val_acc=0.4813 time=293.6s
# [E0] epoch=13 loss=1.6704 train_clip_acc=0.3111 train_song_acc=0.3228 val_acc=0.4813 time=292.7s
# [E0] epoch=14 loss=1.6664 train_clip_acc=0.3093 train_song_acc=0.3212 val_acc=0.5238 time=291.8s
# [E0] epoch=15 loss=1.6563 train_clip_acc=0.3169 train_song_acc=0.3334 val_acc=0.5288 time=290.5s
# [E0] epoch=16 loss=1.6575 train_clip_acc=0.3115 train_song_acc=0.3291 val_acc=0.4713 time=288.2s
# [E0] epoch=17 loss=1.6398 train_clip_acc=0.3172 train_song_acc=0.3322 val_acc=0.5238 time=290.0s
# [E0] epoch=18 loss=1.6512 train_clip_acc=0.3087 train_song_acc=0.3286 val_acc=0.4788 time=287.9s
# [E0] epoch=19 loss=1.6436 train_clip_acc=0.3102 train_song_acc=0.3259 val_acc=0.4813 time=288.5s
# [E0] epoch=20 loss=1.6403 train_clip_acc=0.2912 train_song_acc=0.3059 val_acc=0.5162 time=288.3s
# [E0] epoch=21 loss=1.6398 train_clip_acc=0.3233 train_song_acc=0.3386 val_acc=0.4575 time=287.7s
# [E0] epoch=22 loss=1.6389 train_clip_acc=0.3201 train_song_acc=0.3420 val_acc=0.5100 time=288.1s
# [E0] epoch=23 loss=1.6123 train_clip_acc=0.3264 train_song_acc=0.3406 val_acc=0.5275 time=287.5s
# Early stop at epoch 23, best_val_acc=0.5288
# c:\Code\python\vscode\music_genre_classifier\src\train.py:134: UserWarning: color is redundantly defined by the 'color' keyword argument and the fmt string "0.326393" (-> color=(0.326393, 0.326393, 0.326393, 1.0)). The keyword argument will take precedence.
#   ax3.plot(epochs, train_acc, color='green', label='Train Clip Acc', linestyle='--')
# c:\Code\python\vscode\music_genre_classifier\src\train.py:140: UserWarning: color is redundantly defined by the 'color' keyword argument and the fmt string "0.340639" (-> color=(0.340639, 0.340639, 0.340639, 1.0)). The keyword argument will take precedence.
#   ax4.plot(epochs, train_song_acc, color='orange', label='Train Song Acc', linestyle='--')