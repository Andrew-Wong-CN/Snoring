# Snoring

**Snoring** is a deep learning system designed for **Non-Contact Sleep Staging**. 

So, What is Non-Contact Sleep Staging?

Sleep Staging is a method to diagnose sleep health, specifically, tagging your sleep with five labels(N1, N2, N3, WK, REM) for every 30s. Clinically, Sleep Staging can be implemented with the help of EEG monitoring, connecting the wires to the patient’s head, which is extremely complicated and time-consuming. Consequently, we design a non-contact method based on the breath sound of people during their sleep. Specifically, the microphone on our mobile phones records the sound during people’s sleep, then our deep learning system **Snoring** analyzing the sounds, enhancing the breath sounds in the audio, extract the features of breath sounds, and finally classify them into five sleep stages. 

Sounds Good! 😝But, is that really accurate? 

Definity! We achieve 75% of accuracy on our test data! We find that, in a relatively quiet environment during night, our system **Snoring** can effectively extract the breathing features, especially the **breathing period** and **snore sound**. Through our experiments, we find that the breathing period is much shorter in deep sleep stages while longer in shallow sleep stages. Snore sound usually appears at the junction of deep sleep stages and shallow sleep stages. So, it is explainable that we can extract useful features on the breath sound and classify them to a range of sleep stages by deep learning systems! 

By the way, we call this deep learning system **Snoring** because snore sound is the most significant features for our system. 😲

Then, I will introduce some technical details about this project, **Snoring**.

**Dataset-SnoringNet**

**SnoringNet** is a **totally open-source** corpus for non-contact sleep staging!

We recorded, selected, labeled, and processed 134-hour recordings of breath sounds during sleep with sleep stage labels. We recorded the breath sounds in hospital when patients were taking formal sleep staging monitor. We used mobile phones located on the side of patients to record their breath sounds during their sleeping. 

<p>
![image-20221203101350860](https://github.com/Ameixa551/Snoring/blob/master/images/device_place.png)

</p>


<p>
![image-20221203101350860](https://github.com/Ameixa551/Snoring/blob/master/images/image-20221203101350860.png)

</p>

<p>

![image-20221203101444833](https://github.com/Ameixa551/Snoring/blob/master/images/image-20221203101444833.png)

</p>

<p>

![image-20221203101502637](https://github.com/Ameixa551/Snoring/blob/master/images/image-20221203101502637.png)

</p>