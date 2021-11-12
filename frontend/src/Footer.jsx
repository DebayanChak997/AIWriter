import React from "react";
import fb from "../src/Image/fb.png";
import insta from "../src/Image/insta.png";
import twitter from "../src/Image/twitter.jpg";

const Footer = () => {
  return (
    <>
      {/* <footer className="w-100 bg-light text-center">
        <p>© 2020 IEMA R AND D. All Rights Reserved | Terms and Conditions</p>
      </footer> */}

      <footer className="footer">
  	 <div className="container1">
  	 	<div className="row1">
  	 		<div className="footer-col">
  	 			<h4>Sitesoch</h4>
  	 			<ul>
  	 				<li><a href="#">Home</a></li>
  	 				<li><a href="#">About</a></li>
  	 				<li><a href="#">Pricing</a></li>
  	 				<li><a href="#">Privacy Policy</a></li>
  	 			</ul>
  	 		</div>
  	 		<div className="footer-col">
  	 			<h4>Get Help</h4>
  	 			<ul>
  	 				<li><a href="#">Contact</a></li>
  	 				<li><a href="#">About Service</a></li>
  	 				<li><a href="#">Payment Status</a></li>
  	 				<li><a href="#">Payment Options</a></li>
  	 			</ul>
  	 		</div>
  	 		<div className="footer-col">
  	 			<h4>Address</h4>
  	 			<ul>
  	 				<li><a href="#">6408 Elizabeth Avenue SE, Auburn WA 98092, USA </a></li>
  	 			</ul>
  	 		</div>
  	 		<div className="footer-col">
  	 			<h4 className="footer-h4">follow us</h4>
  	 			<div className="social-links">
  	 				<a href="#">
					   {/* <i className="fab fa-google"></i> */}
					   <img className="imgfoot" src={fb}></img>
					   </a>
  	 				<a href="#">
					   {/* <i className="fab fa-facebook-f"></i> */}
					   <img className="imgfoot" src={insta}></img>
					   </a>
  	 				<a href="#">
					   {/* <i className="fab fa-instagram"></i> */}
					   <img className="imgfoot" src={twitter}></img>
					   </a>
  	 				<a href="#"><i className="fab fa-twitter"></i></a>
  	 			</div>
  	 		</div>
  	 	</div>
  	 </div>
  </footer>
    </>
  );
};

export default Footer;
