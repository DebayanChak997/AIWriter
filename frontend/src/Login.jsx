import React, { useState } from "react";
// import log from "../src/Image/login3.png";
import "./index.css";
import {Animated} from "react-animated-css";
import google from "../src/Image/g-icon.png";
import Navbar from "./Navbar";
import bglog from "../src/Image/login.gif"

import fire from "./fire";
import { userLogin } from "./redux/actions/userActions";
import { useDispatch, useSelector } from "react-redux";
import { useHistory } from "react-router";
import swal from 'sweetalert2';


// const Login = () =>{
//     const [name, setName] = useState();

//     const inputEvent =(event) =>{
//         console.log(event.target.value);
//         setName(event.target.value);
//     }
function Login() {
  const history = useHistory();
  const dispatch = useDispatch();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");


  const {userDetails} = useSelector((state)=> state.userLogin);
  console.log(userDetails);

  // fire.database().ref('users').once('value', function(snapshot22) => {
  //   console.log(snapshot22.val())})
  const onSubmit = async (e) => {
      //e.preventDefault();
      console.log(email, password)


      try {
          fire.auth().signInWithEmailAndPassword(email, password) .then((userCredential) => {
              // Signed in
              const currUser = fire.auth().currentUser
              console.log("user details",currUser)
              const userInfo = {
                  id:currUser.uid,
                  email:currUser.email
              }
              // window.sessionStorage.setItem("user", JSON.stringify(userInfo));
              setTimeout(function(){
                  console.log("go");
                  if(currUser){
                      history.go(0)
                  }
              },1000)
              dispatch(userLogin(userInfo));

                      // check if email is verified
      if (currUser.emailVerified) {
          var user = userCredential.user
          console.log("UserInfo :",user);

      }else {
          // return fire.auth
          //   .signOut()
          //   .then(() => {
          //     // Sign-out successful.
          //     console.log('Signed Out as User is not Verified')
          //     return 'Signed out as User is not Verified'
          //   })
      }
      }) }
      catch (err) {

          console.log(err);

      }
      fire.auth().signInWithEmailAndPassword(email, password).then((u) => {
      console.log(u)
      history.go('/ai')
      })
      .catch((err) => {
        swal.fire({
          icon: 'error',
          title: 'Oops...',
          text: 'Please Enter Valid UserEmail and Password',
        })
          console.log(err);
      })
      // fire.database().ref('users').once('value', (snapshot) => {
      // console.log(snapshot.val());

      // event.preventDefault();
      setEmail("");
      setPassword("");
      // history.go('/welcome')
  };

// const Login = () =>{
//     const [name, setName] = useState();

//     const inputEvent =(event) =>{
//         console.log(event.target.value);
//         setName(event.target.value);
//     }
    return(
        <>
        <Navbar/>
        <Animated animationIn="bounceInDown" animationOut="bounceInOut" isVisible={true}>
        <section className="logsec">


        <div className="wrapper">
  <div className="container2">
    <div className="col-left">
      <div className="login-text">
        <h2>WELCOME</h2>
        <h3>
           <br/> We are happy to have you as a user
        </h3>
        <a className="btn1" href="./register">SIGN UP</a>
      </div>
    </div>
    <div className="col-right">
      <div className="login-form">
        <h2>Login</h2>
        <form>
          <p>
            <input type="text" placeholder="User email" required value={email} onChange={(e) => setEmail(e.target.value)}  />
          </p>
          <p>
            <input type="password" placeholder="Password" required onChange={(e) => setPassword(e.target.value)}/>
          </p>
          <p>
            <input onClick={onSubmit} className="btn1" type="submit" value="Sign In" />
          </p>
{/*           <p> */}
{/*             -----------------OR---------------- */}
{/*           </p> */}
{/*           <p> */}
{/*             Sign In With Google <img className="imglog" src={google}></img> */}
{/*           </p> */}
{/*           <p> */}
{/*             <a href="/reset">Forget password?</a> */}
{/*             */}
{/*           </p> */}
        </form>
      </div>
    </div>
  </div>
  </div>



</section>
</Animated>

        {/* <div className="logdiv">
       <h1 className="logh1">Welcome { name}</h1>
       <input className="loginput" type="text" placeholder="enter your name" onChange={inputEvent} />
       <input className="loginput" type="password" placeholder="enter your password" />
       <button className="logbtn">click me</button>
       <h5 className="logh3"><a href="./register">register here</a></h5>
       </div> */}
        </>

    );

}

export default Login;
