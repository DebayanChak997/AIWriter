import React from "react";
import Navbar from "./Navbar";
import fire, {auth} from "./fire";
import { useForm } from "react-hook-form";
import $ from "jquery";
import swal from 'sweetalert2';
import { useHistory } from "react-router";


const ResetConfirm = () =>{
    const {register, handleSubmit, formState: { errors } } = useForm() ;
    const [email, setEmail] = React.useState("");
    const history = useHistory();
    

    // fire.auth().sendPasswordResetEmail(email).then(function() {
    //     // Email sent.
    //   }).catch(function(error) {
    //     // An error happened.
    //   });
    function getParameterByName(name, url) {
        if (!url) url = window.location.href;
        name = name.replace(/[\[\]]/g, "\\$&");
        var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
            results = regex.exec(url);
        if (!results) return null;
        if (!results[2]) return '';
        return decodeURIComponent(results[2].replace(/\+/g, " "));
    }
    const [mode, setmode] = React.useState("");
    const [actionCode, setactionCode] = React.useState("");
    document.addEventListener('DOMContentLoaded', function() {
        setmode(getParameterByName('mode'));
        setactionCode(getParameterByName('oobCode'));
    });

    console.log(mode);
    console.log(actionCode);
    const [pass, setpass] = React.useState("");
    const handleResetPassword = () => {
        auth.confirmPasswordReset(actionCode, pass).then(function(resp) {
            // Password reset has been confirmed
            console.log(resp);
        }).catch(function(error) {
            // Error occurred during confirmation. The code might have

            // expired or the password is too weak.
            console.log(error);
        });
    }

    $("#btn1").click(function(){
    
        swal.fire({
          title: 'Your Password has been successfully reset! Click on Yes to Login!',
          showDenyButton: true,
          confirmButtonText: 'Yes',
          denyButtonText: 'No',
          
        }).then((result) => {
          if (result.isConfirmed) {
            
            window.location.reload();
          } else if (result.isDenied) {
            history.push('/login')
          }
        })
      })

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
    <form className="login-form" onSubmit={handleSubmit(handleResetPassword)}>
      {/* <input className="resetinput" required value={email} onChange={(e) => setEmail(e.target.value)} /> */}
      <input className="resetinput" type="password" required value={pass} onChange={(e) => setpass(e.target.value)} placeholder="New Password"/>
      <input className="resetinput" type="password" placeholder="Retype Password"/>
      <input id="btn1" className="btn1" type="submit" value="Reset Password" />
   
    </form>
  </div>
</div>
</body>
        </>
    )
}

export default ResetConfirm;