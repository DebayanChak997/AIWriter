import React from "react";
import Navbar from "./Navbar";

const Payment = () =>{
 return(
     <>
   <Navbar/>
  
<body className="paybody">
<div className="card">

<div className="top">
    <div className="payment"></div>
 
    </div>

  
  <div className="card-body1">
  <h1 className="payh1">PAYMENT</h1>
    
    <form className="payform">
      <div className="one">
      <label for=''>Name on card</label>
      <input className="payinput" placeholder='Name' type='text' />
      </div>
      
      <div className="two">
      <label for=''>Card Number</label>
      <input className="payinput" maxlength='16' placeholder='Card Number' type='number' />
      </div>
      
      <div className="three">
      <label for=''>Expiry Date</label>
      <select>
        <option>January</option>
        <option>February</option>
        <option>March</option>
        <option>April</option><option>May</option>
        <option>June</option>
        <option>July</option>
        <option>August</option>
        <option>September</option><option>October</option>
        <option>November</option>
        <option>December</option>
      </select>
      </div>
      
      <div className="four">
      
        <label for=''>&#x200b;</label>
      <select>
        <option>2015</option>
        <option>2016</option>
        <option>2017</option>
        <option>2018</option>
        <option>2019</option>
        <option>2020</option>
      </select>
      </div>
      <div className="five">
      <label for=''>CVV</label>
      <input className="payinput" maxlength='3' placeholder='633' type='number' />
      </div>
      <div > <button className="paybutton">SUBMIT</button></div>
    </form>
    
  </div>
</div>
</body>

</>
)
}

export default Payment;