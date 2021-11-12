import React, { useState } from "react";
import {Animated} from "react-animated-css";
import contact from "../src/Image/contactus.gif";
import ScrollAnimation from 'react-animate-on-scroll';



const Contact = () => {
  // const [data, setData] = useState({
  //   fullname: "",
  //   phone: "",
  //   email: "",
  //   msg: "",
  // });

  // const inputEvent = (event) => {
  //   const { name, value } = event.target;

  //   setData((preVal) => {
  //     return {
  //       ...preVal,
  //       [name]: value,
  //     };
  //   });
  // };

  // const formSubmit = (e) => {
  //   e.preventDefault();
  //   alert(
  //     `My name is ${data.fullname}. My mobile number is ${data.phone} and email is ${data.email}, Here is what I watn to say ${data.msg}`
  //   );
  // };

  return (
    <>
   
   {/* <Animated animationIn="bounceInDown" animationOut="bounceOut" isVisible={true}> */}
   <ScrollAnimation animateIn="bounceInLeft">

  <section class="contact">
    <div class="content">
      <h2>Contact Us </h2>
      <p>Let us know how can we help.</p>
    </div>
    <div class="container">
      <div class="contactform">
        <form action="javascript:sendmail()" method="post">
          <h2>Send Message </h2>
          <div class="inputBox">
            <input id="Name" type="text" name="" required="required"/>
            <span>Name</span>
          </div>
          <div class="inputBox">
            <input id="Email" type="text" name="" required="required"/>
            <span>Email</span>
          </div>
          <div class="inputBox">

            <textarea id="Message" required="required"></textarea>
            <span>Message</span>
          </div>
          <div class="inputBox">
            <input type="submit" name="" value="Send"/>
          </div>
        </form>
      </div>
      
      <img className="contactimg" src={contact}></img>
    </div>
  </section>
  </ScrollAnimation>
 {/* </Animated> */}

    </>
  );
};

export default Contact;
