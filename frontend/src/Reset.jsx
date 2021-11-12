import React from "react";
import Navbar from "./Navbar";
import fire from "./fire";
import $ from "jquery";
import swal from 'sweetalert2';
import { useHistory } from "react-router";
import { useForm } from "react-hook-form";


const Reset = () =>{
    const [email, setEmail] = React.useState("");
    
    const {register, handleSubmit, formState: { errors } } = useForm() ;
    const history = useHistory();
    
    fire.auth().sendPasswordResetEmail(email).then(function() {
        // Email sent.
      }).catch(function(error) {
        // An error happened.
      });



    return(
        <>
        <Navbar/>
        <body className="resetsec">
<div className="login-page5">
<h1 className="reseth1">RESET PASSWORD</h1>
  <div className="form5">
    <form className="register-form">
      <input className="resetinput" type="text" placeholder="name"/>
      <input className="resetinput" type="password" placeholder="password"/>
      <input className="resetinput" type="password" placeholder="password"/>
      
      <input className="resetinput" type="text" placeholder="email address"/>
      <button>create</button>
      <p className="message">Already registered? <a href="#">Sign In</a></p>
    </form>
    <form className="login-form">
      <input id="resetinput" className="resetinput" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email Address"/>
      {/* <input className="resetinput" type="password" placeholder="New Password"/> */}
      {/* <input className="resetinput" type="password" placeholder="Retype Password"/> */}
      <button>Send Reset Link</button>
   
    </form>
  </div>
</div>
</body>
        </>
    )
}

export default Reset;