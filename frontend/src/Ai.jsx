import React,{useState}  from "react";
import icon from "../src/Image/icon.png";
import "../node_modules/bootstrap/dist/css/bootstrap.min.css";
import "../node_modules/bootstrap/dist/js/bootstrap.bundle";
import iconai from "../src/Image/iconai.png";
import { useForm } from "react-hook-form";
import axios from "axios";
import swal from 'sweetalert2';
import { useHistory } from "react-router";
//redux import
import { userLogout } from "./redux/actions/userActions";
import { useDispatch, useSelector } from "react-redux"
import $ from "jquery";

import fire from "./fire";
var displaynn;
var disname;
var dummyData;

const Ai =()=>{
    var [coins, setcoins] = useState(0);
    var name ;
    var name2;
    var [uname, setuname] = useState("");
    const {register, handleSubmit, formState: { errors } } = useForm() ;
    const onSubmit = data => axios.post(
            "http://127.0.0.1:8004/api/handleaidata/",
            data
            ) 
            .then(res => console.log(res));
            const dispatch = useDispatch();
            const history = useHistory()
            const { userDetails } = useSelector((state) => state.userLogin);
            const [numOfcoins, SetNoOfCoins] = useState("")
            console.log(userDetails);
          
          
            var email = userDetails.email;
            console.log(email);
            fire.database().ref('users').child('userdetails').orderByChild('useremail').equalTo(email).once('value', 
            function(Allrecords) {
                console.log(Allrecords.val())
                Allrecords.forEach(function(childSnapshot) {
                    var childData = childSnapshot.val();
                    var wc = Number(childData.writers_coin);
                    setcoins(wc);
                    name = childData.username;
                    name2 = name;
                    setuname(name2);
                    console.log(coins);
                });
            });
          
          //   function openForm() {
          //     if(coins > 0){
          //       coins = coins - 1;
          //     fire.database().ref("users/userdetails"+"/"+name+"/writers_coin").set(coins);
          //     <form action="/ai" className="formContainer">
          //     <h4>Writer Coins: {coins}</h4>
          //     <button type="submit" className="btn">YES</button>
          //     {/* <button type="button" className="btn cancel" onClick={closeForm}>Close</button> */}
          //     </form>
          //                     document.getElementById("popupForm").style.display = "block";
                            
          //     }
          //     else{
          //       alert("You don't have enough coins");
          //       <h4>Writer Coins: {coins}</h4>
          //     }
          //   }
          
          // function closeForm() {
          //                 document.getElementById("popupForm").style.display = "none";
          //               }
          // //   const onSubmit = data => axios.post(
          // //     "http://127.0.0.1:8004/api/handleaidata/",
          // //     data
          // //     ) 
          // //     .then(res => console.log(res));
          
          
          $("#btnSubmit").click(function(){
              
              swal.fire({
                title: 'The AI Generated Output Will be Mailed Within Few Minutes To The Provided Mail ID.Do you want to give another input?',
                showDenyButton: true,
                confirmButtonText: 'Yes',
                denyButtonText: 'No',
                
              }).then((result) => {
                if (result.isConfirmed) {
                  if(coins > 0){
                      coins = coins - 1;
                    fire.database().ref("users/userdetails"+"/"+name+"/writers_coin").set(coins);}
                  window.location.reload();
                } else if (result.isDenied) {
                  history.push('/')
                }
              })
            })
          
          
            function logoutHandler() {
              dispatch(userLogout())
              if (dispatch) {
                history.go(0)
              }
          
              setTimeout(
                function () {
                  history.go(0)
                },
                1000
              )
            }
          
            if (userDetails) {
              var email22 = userDetails.email;
          
              fire.database().ref('users').child('userdetails').orderByChild('useremail').equalTo(email22).once('value', function (Allrecords) {
                // console.log(Allrecords);
                dummyData = JSON.stringify(Allrecords.val())
                console.log(JSON.stringify(Allrecords.val()));
          
                Allrecords.forEach(function (childSnapshot) {
                  disname = childSnapshot.val();
                  displaynn = disname.username;
                  SetNoOfCoins(displaynn)
                  console.log(displaynn)
                });
              })
            }
          
                    
    return (

        <>
<body id="page-top">

<div id="wrapper">

    
    <ul className="navbar-nav bg-gradient-primary sidebar sidebar-dark accordion" id="accordionSidebar">

   
        <a className="sidebar-brand d-flex align-items-center justify-content-center" href="/">
            <div className="sidebar-brand-icon rotate-n-15">
                <i className="fas fa-laugh-wink"></i>
            </div>
            <div className="sidebar-brand-text mx-3">IEM AI WRITER</div>
        </a>

    
        <hr className="sidebar-divider my-0"/>

        
        <li className="nav-item">
            <a className="nav-link1" href="/">
                <i className="fas fa-fw fa-tachometer-alt"></i>
                <span>Home</span></a>
        </li>

       
        <hr className="sidebar-divider"/>

        <div className="sidebar-heading">
            Interface
        </div>

       
        <li className="nav-item">
            <a className="nav-link1 collapsed" href="#" data-toggle="collapse" data-target="#collapseTwo"
                aria-expanded="true" aria-controls="collapseTwo">
                <i className="fas fa-fw fa-cog"></i>
                <span>Service</span>
            </a>
            {/* <div id="collapseTwo" className="collapse" aria-labelledby="headingTwo" data-parent="#accordionSidebar">
                <div className="bg-white py-2 collapse-inner rounded">
                    <h6 className="collapse-header">Custom Components:</h6>
                    <a className="collapse-item" href="buttons.html">Buttons</a>
                    <a className="collapse-item" href="cards.html">Cards</a>
                </div>
            </div> */}
        </li>

        <li className="nav-item active">
            <a className="nav-link1" href="#" data-toggle="collapse" data-target="#collapseUtilities"
                aria-expanded="true" aria-controls="collapseUtilities">
                <i className="fas fa-fw fa-wrench"></i>
                <span>Contact</span>
            </a>
            {/* <div id="collapseUtilities" className="collapse show" aria-labelledby="headingUtilities"
                data-parent="#accordionSidebar">
                <div className="bg-white py-2 collapse-inner rounded">
                    <h6 className="collapse-header">Custom Utilities:</h6>
                    <a className="collapse-item" href="utilities-color.html">Colors</a>
                    <a className="collapse-item active" href="utilities-border.html">Borders</a>
                    <a className="collapse-item" href="utilities-animation.html">Animations</a>
                    <a className="collapse-item" href="utilities-other.html">Other</a>
                </div>
            </div> */}
        </li>

  
        {/* <hr className="sidebar-divider"/>

        <div className="sidebar-heading">
            Addons
        </div>

    
        <li className="nav-item">
            <a className="nav-link1 collapsed" href="#" data-toggle="collapse" data-target="#collapsePages"
                aria-expanded="true" aria-controls="collapsePages">
                <i className="fas fa-fw fa-folder"></i>
                <span>Pages</span>
            </a>
            <div id="collapsePages" className="collapse" aria-labelledby="headingPages" data-parent="#accordionSidebar">
                <div className="bg-white py-2 collapse-inner rounded">
                    <h6 className="collapse-header">Login Screens:</h6>
                    <a className="collapse-item" href="login.html">Login</a>
                    <a className="collapse-item" href="register.html">Register</a>
                    <a className="collapse-item" href="forgot-password.html">Forgot Password</a>
                    <div className="collapse-divider"></div>
                    <h6 className="collapse-header">Other Pages:</h6>
                    <a className="collapse-item" href="404.html">404 Page</a>
                    <a className="collapse-item" href="blank.html">Blank Page</a>
                </div>
            </div>
        </li>

       
        <li className="nav-item">
            <a className="nav-link1" href="charts.html">
                <i className="fas fa-fw fa-chart-area"></i>
                <span>Charts</span></a>
        </li>

       
        <li className="nav-item">
            <a className="nav-link1" href="tables.html">
                <i className="fas fa-fw fa-table"></i>
                <span>Tables</span></a>
        </li>

        
        <hr className="sidebar-divider d-none d-md-block"/>

       
        <div className="text-center d-none d-md-inline">
            <button className="rounded-circle border-0" id="sidebarToggle"></button>
        </div> */}

    </ul>
    

  
    <div id="content-wrapper" className="d-flex flex-column">

       
        <div id="content">

           
            <nav className="navbar navbar-expand navbar-light bg-white topbar mb-4 static-top shadow">

              
                <button id="sidebarToggleTop" className="btn btn-link d-md-none rounded-circle mr-3">
                    <i className="fa fa-bars"></i>
                </button>

               
                
               
                <ul className="navbar-nav ml-auto">

                <span className="mr-2 d-none d-lg-inline text-gray-600 small" style={{marginTop:'25px', fontSize:'15px'}}>Writer's Coin:{coins}</span>
                    <div className="topbar-divider d-none d-sm-block"></div>
                    <div className="topbar-divider d-none d-sm-block"></div>

                  
                    <li className="nav-item dropdown no-arrow">
                        <a className="nav-link1 dropdown-toggle" href="#" id="userDropdown" role="button"
                            data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                            <span className="mr-2 d-none d-lg-inline text-gray-600 small">{numOfcoins}</span>
                            <img className="img-profile rounded-circle"
                                src={icon} />
                        </a>
                       
                        <div className="dropdown-menu dropdown-menu-right shadow animated--grow-in"
                            aria-labelledby="userDropdown">
                            {/* <a className="dropdown-item" href="#">
                                <i className="fas fa-user fa-sm fa-fw mr-2 text-gray-400"></i>
                                Profile
                            </a>
                            <a className="dropdown-item" href="#">
                                <i className="fas fa-cogs fa-sm fa-fw mr-2 text-gray-400"></i>
                                Settings
                            </a>
                            <a className="dropdown-item" href="#">
                                <i className="fas fa-list fa-sm fa-fw mr-2 text-gray-400"></i>
                                Activity Log
                            </a> */}
                            <div className="dropdown-divider"></div>
                            <a className="dropdown-item" href="#" data-toggle="modal" data-target="#logoutModal">
                                <i className="fas fa-sign-out-alt fa-sm fa-fw mr-2 text-gray-400"></i>
                                Logout
                            </a>
                        </div>
                    </li>

                </ul>

            </nav>


 {/* <div className="sliderai">
<p>Temparature</p>
<input type="text" className="inputai" />

</div> */}
          

        

            <div className="container-fluid">

              
                <h1 className="h3 mb-1 text-gray-800">IEM AI WRITER</h1>
          

              
                <div className="row"  style={{width:"120%"}}>
                 <div className="col-xl-8 col-lg-7">
                        <div className="cardai shadow mb-4">
                          
                            <div className="cardai-header py-3 d-flex flex-row align-items-center justify-content-between" >
                                
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="form-group" >
          <label for="exampleInputEmail1">Title</label>

          <img  src={iconai} alt="#" class="instruction"/>
         <div className="hide">This component signifies, The Generation of AI Content, Justifiable with 10-25 Characters.</div>

          <input {...register("title")} style={{width:"97%" }} type="text" className="form-control"  aria-describedby="emailHelp"  />
          
          </div>
          <div className="form-group">
          <label for="exampleInputPassword1">Sub-Title</label>

          <img  src={iconai} alt="#" class="instruction"/>
          <div className="hide">It signifies the motive of the blog to be genarated.</div>

          <input  {...register("subtitle")} style={{width:"97%" }}  type="text" className="form-control"  />
          </div>
          <div className="form-group">
          <label for="exampleInputPassword1">Catagory</label>

          <img  src={iconai} alt="#" class="instruction"/>
          <div className="hide">It is related to both Title and Sub-Title.</div>

          <input  {...register("category")} style={{width:"97%" }}  type="text" className="form-control" />
          </div>
          <div className="form-group">
          <label for="exampleInputPassword1">Sub-Catagory</label>

          <img  src={iconai} alt="#" class="instruction"/>
          <div className="hide">It has a relation with Category, establishing the Sementic Search Capabilities of the Generation.</div>

          <input  {...register("subcategory")} style={{width:"97%" }}  type="text" className="form-control" />
          </div>
          <div className="form-group">
          <label for="exampleInputPassword1">Keyword</label>
          
          <img  src={iconai} alt="#" class="instruction"/>
          <div className="hide">Keywords are for the words that are related to the title, It is comma separated.</div>

          <input  {...register("keyword")} style={{width:"97%" }}  type="text" className="form-control" />
          </div>
          <div className="form-group">
          <label for="exampleInputPassword1">Abstract</label>

          <img  src={iconai} alt="#" class="instruction"/>
          <div className="hide">Abstract is the main part, Justifiable with 100-150 Characters, that must be provided by the user, To govern the randomness of our Generated Blog.</div>

          <textarea  {...register("abstract")} style={{width:"97%", overflow: "scroll" }}  type="text" className="form-control" />
          </div>
          <div className="form-group">
          <label for="exampleInputPassword1">Email</label>

          <img  src={iconai} alt="#" class="instruction"/>
          <div className="hide">Enter your E-Mail ID to get the generated required Blog as output in your E-Mail.</div>

          <input  {...register("email")} style={{width:"97%"  }}  type="email" className="form-control" />
          </div>

          <div className="form-group2">
          <label for="exampleInputPassword1" style={{marginRight:"20px"}}>Temparature</label>
          <input  {...register("temp")} style={{width:"10%" ,marginRight:"20px" }}  type="text" className="form-control" />  
          <label for="exampleInputPassword2" style={{marginRight:"20px"}}>Top K</label>
          <input  {...register("top_k")} style={{width:"10%" ,marginRight:"20px" }}  type="text" className="form-control" />       
          <label for="exampleInputPassword2" style={{marginRight:"20px"}}>Top P</label>
          <input  {...register("top_p")} style={{width:"10%" ,marginRight:"20px" }}  type="text" className="form-control" />       
          </div>

          



         
          
          {coins==0 ? <h4>Please buy more coins!</h4> :<input id="btnSubmit" className="btn btn-primary" type="submit" value="Submit"/>}

          {/* <div className="sliderai">
    
 <div className="divai">
<p>Temparature</p>
<input type="text" className="inputai" required />
</div>

<div className="divai">
<p>Top K</p>
<input type="text" className="inputai" required/>
</div>


<div className="divai">
<p>Top P</p>
<input type="text" className="inputai" required />
</div>

</div> */}

        </form>

        
                           </div>

                           
                           

               </div> 



        </div>
        

    </div>

 </div>

</div>
</div>


<a className="scroll-to-top rounded" href="#page-top">
    <i className="fas fa-angle-up"></i>
</a>


<div className="modal fade" id="logoutModal" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel"
    aria-hidden="true">
    <div className="modal-dialog" role="document">
        <div className="modal-content">
            <div className="modal-header">
                <h5 className="modal-title" id="exampleModalLabel">Ready to Leave?</h5>
                <button className="close" type="button" data-dismiss="modal" aria-label="Close">
                    <span aria-hidden="true">×</span>
                </button>
            </div>
            <div className="modal-body">Select "Logout" below if you are ready to end your current session.</div>
            <div className="modal-footer">
                <button className="btn btn-secondary" type="button" data-dismiss="modal">Cancel</button>
                <a onClick={logoutHandler} className="btn btn-primary" href="/">Logout</a>
            </div>
        </div>
    </div>
</div>
</div>



</body>

        </>


    )

}


export default Ai;