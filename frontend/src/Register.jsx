import React, { useState } from "react";
import "./index.css";
import {Animated} from "react-animated-css";
import Navbar from "./Navbar";
import firebase,{auth} from './fire';
import  {useHistory}  from "react-router";
import swal from 'sweetalert2';

const Register = () =>{
    const [name, setName] = useState();
    const [email, setEmail] = useState();
    const [pass, setPass] = useState();
    const history = useHistory();

    // const inputEvent =(event) =>{
    //     console.log(event.target.value);
    //     setName(event.target.value);
    // }

     // Push Function
//   const Push = (e) => {
//     e.preventDefault();
//     try
//     {
//
//         const authe = firebase.auth().createUserWithEmailAndPassword(email,pass).then((userCredentials)=>{
//            })
//          console.log("checking auth : ",authe)
//         if(authe)
//         {
//          firebase.database().ref("/users/userdetails" + "/" + name).set({
//           username: name,
//           useremail: email,
//           password: pass,
//           writers_coin:3
//           })
//     }
//     catch (err) {
//       swal.fire(
//         'YOUR ACCOUT HAS BEEN CREATED SUCCESSFULLY!',
//         'success'
//       )
//       console.log(err);
//     }
//     history.push('/login')
//   }
const Push = (e) => {
    e.preventDefault();
    try
    {

      const authe = firebase.auth().createUserWithEmailAndPassword(email,pass).then((userCredentials) =>{
        // userCredentials.user.sendEmailVerification();
        // auth.signOut();
      })
      console.log("checking auth : ",authe)
      if(authe){
        firebase.database().ref("/users/userdetails" + "/" + name).set({
          username: name,
          useremail: email,
          password: pass,
          writers_coin:3

      })
      }
    }
    catch (err) {
      console.log(err);
    }
    swal.fire(
      'YOUR ACCOUT HAS BEEN CREATED SUCCESSFULLY!',
      'success'
    )
    history.push('/login')
  }
// const Register = () =>{
//     const [name, setName] = useState();

//     const inputEvent =(event) =>{
//         console.log(event.target.value);
//         setName(event.target.value);
//     }
    return(
        <>
        <Navbar/>
         <Animated animationIn="bounceInDown" animationOut="bounceInOut" isVisible={true}>
        <section className="logsec1">
   

<div className="wrapper">
  <div className="container2">
    <div className="col-left">
      <div className="login-text">
        <h2>HELLO</h2>
        <p>
          {name} 
        </p>
        <h3>HAPPY TO SEE YOU</h3>
       
        <a className="btn1" href="./login">SIGN IN</a>
        
      </div>
    </div>
    <div className="col-right">
      <div className="login-form">
        <h2>Sign Up</h2>
        <form>
          <p>
            <input type="text" placeholder="Username" required value={name} onChange={(e) => setName(e.target.value)}  />
          </p>
          <p>
            <input type="email" placeholder="EmailId" required value={email} onChange={(e) => setEmail(e.target.value)}  />
          </p>
          <p>
            <input type="password" placeholder="Password" value={pass} onChange={(e) => setPass(e.target.value)} required />
          </p>
          <p>
            <input type="password" placeholder="Confirm Password" required />
          </p>
          <p>
            <input onClick={Push} className="btn1" type="submit" value="Sign Up" />
          </p>
          <p>
            <a href="">Forget password?</a>
           
          </p>
        </form>
      </div>
    </div>
  </div>
  
</div>

</section>
</Animated>
        {/* <div className="logdiv">
       <h1 className="logh1">Hello { name}</h1>
       <input className="loginput" type="text" placeholder="enter your name" onChange={inputEvent} />
       <input className="loginput" type="email" placeholder="enter your email" />
       <input className="loginput" type="password" placeholder="enter your password" />
       <input className="loginput" type="password" placeholder="confirm your password" />
       <button className="logbtn">Submit</button>
       <h5 className="logh3"><a href="./login">LogIn here</a></h5>
       </div> */}
        </>

    );

}

export default Register;