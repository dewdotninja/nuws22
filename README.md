## การฝึกอบรมเชิงปฏิบัติการเรื่อง "โครงข่ายประสาทเทียมและการเรียนรู้เชิงลึก"

<p />  
<p align="center">
<img src="https://drive.google.com/uc?id=1gfBnit4FRu-MvGjOGpVEw21TxwzNbq-R" width=500 />
</p>

<img align=left src="https://i.imgur.com/CzEUVpd.jpg" width=250 /> 
<ul>
  <li /><b>ชื่อเรื่อง :</b> โครงข่ายประสาทเทียมและการเรียนรู้เชิงลึก (Artificial Neural Networks and Deep Learning)
  <li /><b>วิทยากร :</b> ดร.วโรดม ตู้จินดา <img align=right src="https://drive.google.com/uc?id=1t6666zvVYo6I0fncdc_XN65Wxino_hTx" width=150 />
  <li /><b>สถานที่ :</b> ภาควิชาฟิสิกส์ คณะวิทยาศาสตร์ มหาวิทยาลัยนเรศวร
  <li /><b>วัน-เวลา :</b> 14 - 15 มิถุนายน 2565
</ul>
<hr>
จากคำบรรยายในวิกิพิเดีย ปัญญาประดิษฐ์ (artificial intelligence) 
หรือที่นิยมเรียกโดยย่อว่า AI คือเชาว์ปัญญาที่แสดงให้เห็นในเครื่องจักรกล 
แตกต่างจากปัญญาธรรมชาติจากสมองของมนุษย์หรือสัตว์ ดังนั้น AI จะครอบคลุมนวัตกรรมทั้งหมดที่ทำให้คอมพิวเตอร์มีความชาญฉลาดเข้าสู่ปัญญาของมนุษย์ 
ซึ่งมีขอบเขตที่กว้างมาก การเรียนรู้ของเครื่อง (machine learning) คือเซตย่อยของเอไอที่เน้นการศึกษาขั้นตอนวิธีทางคอมพิวเตอร์ในการเรียนรู้และปรับตัวจากข้อมูล ซึ่งสามารถทำได้
หลายแนวทางเช่นการหาค่าเหมาะที่สุด ทฤษฏีกราฟ แต่ในการฝึกอบรมครั้งนี้จะเน้นการใช้โครงข่ายประสาทเทียม (artificial neural network) ซึ่งต่อไปจะอ้างถึงโดยตัวย่อ 
ANN ในการเรียนรู้ กล่าวได้ว่า ANN คือระบบคอมพิวเตอร์หรือโมเดลทางคณิตศาสตร์ที่ใช้ในการจำลองสมองทางชีวภาพ โดยอาศัยการเรียนรู้หรือเรียกว่าการฝึก (training) 
โดยข้อมูลตัวอย่าง เมื่อเขียนภาพรวมความสัมพันธ์ของขอบเขตปัญหาที่กล่าวมาจะได้ดังแสดงในรูป 

<p />  
<p align="center">
<img src="https://drive.google.com/uc?id=1r_T5zq9MMcGXptpF_k1BU5nfC4Kq6WM_" width=400 />
</p>
<div align="center">ความสัมพันธ์ของปัญญาประดิษฐ์ การเรียนรู้ของเครื่อง และการเรียนรู้เชิงลึก</div>

### โครงข่ายประสาทเทียมและการเรียนรู้เชิงลึก

มักมีความเข้าใจผิดกับคำว่า "เชิงลึก" ในการเรียนรู้เชิงลึก ซึ่งมิได้มีความหมายในเชิงปรัชญา หรือแปลว่าการเรียนรู้อย่างถ่องแท้แต่อย่างใด แต่เป็นคำคุณศัพท์ที่ขยายความ เกี่ยวกับจำนวนชั้นของโครงข่ายประสาทเทียม โดย ANN ที่มีเฉพาะชั้นที่เป็นอินพุตและเอาต์พุตเรียกว่าเป็นแบบตื้น (shallow) ส่วนโครงข่ายประสาทเทียมเชิงลึก (deep neural network) ซึ่งต่อไปจะเรียกโดยย่อว่า DNN นอกจากชั้นอินพุตและเอาต์พุตแล้วยังประกอบด้วยชั้นแฝง (hidden layer) อย่างน้อยหนึ่งชั้น 

สถาปัตยกรรมของโครงข่ายประสาทเทียมที่ใช้ในการเรียนรู้เชิงลึกมีจำแนกได้เป็นหลายประเภทขึ้นกับการใช้งาน ในการฝึกอบรมครั้งนี้จะศึกษาตั้งแต่โมเดล DNN ขั้นพื้นฐาน 
หลักการของการเรียนรู้โดยปรับค่าเกรเดียนต์ของค่าสูญเสียในทิศทางเข้าสู่ค่าต่ำสุด ปัญหาการฟิตเกิน (overfitting) และการแก้ไข หลักการของตัวหาค่าเหมาะที่สุด 
(optimizers) ประเภทต่างๆ โมเดล CNN (Convolutional Neural Networks) สำหรับงานที่เกี่ยวข้องกับข้อมูลภาพ การถ่ายโอนการเรียนรู้ (transfer 
learning) และโมเดล RNN (Recurrent Neural Networks), GRU (Gated Recurrent Unit) , LSTM (Long Short Term Memory) สำหรับข้อมูลลำดับ เช่นงานด้าน NLP (Natural Language Processing) การพยากรณ์ข้อมูลฐานเวลา 

<p />  
<p align="center">
<img src="https://drive.google.com/uc?id=1ccOQjyevWY2bErjs-nIevtGjCVlpHNwA" width=600 />
</p>

<hr>

### หมายกำหนดการ

#### อังคาร 14 มิถุนายน 2565

9 AM - 12 PM

* ติดตั้งซอฟต์แวร์และการใช้ Jupyter notebook
* ปัญญาประดิษฐ์ การเรียนรู้ของเครื่อง และการเรียนรู้เชิงลึก
* หลักการของการเรียนรู้โดยมีผู้สอน
* การเขียนโปรแกรม Python และใช้งาน tensorflow เบื้องต้น
* โมเดลการถดถอยเชิงเส้น (linear regression) โดยโครงข่ายประสาทเทียม
* โมเดลการถดถอยลอจิสติก (logistic regression)
* ฟังก์ชันสูญเสีย (loss functions)

1 - 4 PM

* แผนภาพเชิงคำนวณ (computational graph)
* โครงข่ายประสาทเทียมเชิงลึก
* การกำหนดค่าเริ่มต้นพารามิเตอร์เรียนรู้
* การทำเรกูลาไรเซชัน (regularizations) และดรอปเอาต์ (dropouts)
* ตัวหาค่าเหมาะที่สุด (optimizers)
* การทำกลุ่มให้เป็นบรรทัดฐาน (batch normalization)
* ช่วงปฏิบัติ

#### พุธ 15 มิถุนายน 2565

9 AM - 12 PM

* การดำเนินการสังวัตนาการในการประมวลผลภาพ
* พื้นฐานของโครงข่ายประสาทเทียมเชิงสังวัตนาการ (CNN : comvolutional neural networks)
* ชั้นพูลลิง (pooling layer)
* ตัวอย่างโมเดล CNN : LeNet-5, AlexNet, VGG-16, VGG-19, ResNets
* โมเดลอินเซปชัน (inception model)
* การถ่ายโอนการเรียนรู้ (transfer learning)
* ช่วงปฏิบัติ

1 - 4 PM

* สถาปัตยกรรมโครงข่ายประสาทเทียมวกกลับ (RNN : recurrent neural networks)
* โมเดล GRU (gated recurrent unit)
* โมเดล LSTM (long short term memory)
* องค์ประกอบข้อมูลอนุกรมเวลา
* การพยากรณ์อนุกรมเวลา
* ช่วงปฏิบัติ
* สรุปการอบรมและตอบคำถาม


<hr>

### อุปกรณ์ที่ใช้ในการอบรม

<ul>
  <li />เครื่องคอมพิวเตอร์ระบบปฏิบัติการ Windows, Mac-OSX หรือ Linux (ซอฟต์แวร์ที่ใช้สามารถใช้งานฟรีทั้งหมด ควรติดตั้งก่อนการอบรม)
  <li />สัญญาณ WiFi เพื่อเชื่อมต่ออินเทอร์เน็ต 
</ul>

### ซอฟต์แวร์และการติดตั้ง

ผู้ฝึกอบรมสามารถเลือกติดตั้งซอฟต์แวร์ลงบนเครื่องตามคำแนะนำในภาคผนวก B ด้านล่าง หรือใช้ Google colab 

<a href="https://github.com/dewdotninja/books/blob/main/th/anndl/appendixB.ipynb">ภาคผนวก B : การติดตั้งซอฟต์แวร์</a>

### ลิงก์ภายนอก

<ul>
<li /><a href="https://github.com/amanchadha/coursera-deep-learning-specialization">Coursera Deep Learning Specialization</a>
<li /><a href="https://github.com/lmoroney/dlaicourse">Laurence Moroney dlai course github</a>
</ul>

### ลิงก์จาก google drive
<ul>
    <li /><a href="https://drive.google.com/drive/folders/1Q59m3dwZFlSdR-RjqynpXeAmaU4vMIAj">datasets</a>
  <li /><a href="https://drive.google.com/drive/folders/16u6qkCrGrMRHEjGXLDD3iUxO4BV8QZ-i">Exercises</a>
</ul>
